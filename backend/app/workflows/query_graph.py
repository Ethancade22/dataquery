from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..config import Settings, settings
from ..database import SCHEMA
from ..errors import PipelineStageError
from ..mcp_runtime import LocalMcpClient, create_local_mcp_server
from ..model_client import ModelClient
from ..models import QueryResult
from ..preprocessing import RequestPreprocessor
from ..querying.duckdb_engine import DuckDbEngine
from ..querying.data_qa_agent import DataQaAgent
from ..querying.models import SqlExecution
from ..querying.response_generator import ResponseGenerator
from ..querying.single_database_agent import SingleDatabaseAgent
from ..retrieval import SchemaGraphBuilder, SchemaIndex
from ..security import AccessScope
from ..semantic import (
    BusinessSemanticLayer,
    CorrectionRouter,
    SemanticLLMJudge,
    SemanticPlanner,
    SemanticQueryPlan,
    SemanticValidationIssue,
    SemanticValidator,
)
from ..skills import SkillRegistry
from .result_builder import ResultBuilder
from .state import QueryState


class QueryWorkflow:
    """基于 LangGraph 的问答和数据库查询工作流。"""

    def __init__(
        self,
        model_client: ModelClient,
        schema_index: SchemaIndex,
        config: Settings | None = None,
    ) -> None:
        self.model_client = model_client
        self.schema_index = schema_index
        self.config = config or settings
        self.preprocessor = RequestPreprocessor(model_client)
        self.skills = SkillRegistry()
        self.graph_builder = SchemaGraphBuilder()
        self.database_engine = DuckDbEngine()
        self.single_database_agent = SingleDatabaseAgent(
            model_client,
            self.mcp_client,
            self.skills.get("database_query"),
            self.config.mcp_max_tool_calls,
        )
        self.response_generator = ResponseGenerator(
            model_client,
            self.config,
        )
        self.business_semantics = BusinessSemanticLayer()
        self.semantic_planner = SemanticPlanner(self.business_semantics)
        self.semantic_validator = SemanticValidator(self.business_semantics)
        self.semantic_judge = SemanticLLMJudge(model_client)
        self.correction_router = CorrectionRouter()
        self.data_qa_agent = DataQaAgent(
            model_client,
            self.mcp_client,
            self.skills.get("data_qa"),
        )
        self.checkpointer = InMemorySaver()
        self.graph = self._compile()

    def mcp_client(self, access_scope: dict[str, Any]) -> LocalMcpClient:
        scope = AccessScope.from_dict(access_scope)
        return LocalMcpClient(create_local_mcp_server(self.database_engine, scope))

    def _compile(self):
        builder = StateGraph(QueryState)
        builder.add_node("preprocess", self._preprocess)
        builder.add_node("respond_directly", self._respond_directly)
        builder.add_node("answer_qa", self._answer_qa)
        builder.add_node("retrieve_schema", self._retrieve_schema)
        builder.add_node("build_semantic_plan", self._build_semantic_plan)
        builder.add_node("human_clarification", self._human_clarification)
        builder.add_node("prepare_single_database", self._prepare_single_database)
        builder.add_node("execute_single_database", self._execute_single_database)
        builder.add_node("run_multi_database", self._run_multi_database)
        builder.add_edge(START, "preprocess")
        builder.add_conditional_edges(
            "preprocess",
            lambda state: {
                "direct_response": "respond_directly",
                "data_qa": "answer_qa",
                "database_query": "retrieve_schema",
            }[state["intent"]["action"]],
            {
                "respond_directly": "respond_directly",
                "answer_qa": "answer_qa",
                "retrieve_schema": "retrieve_schema",
            },
        )
        builder.add_edge("respond_directly", END)
        builder.add_edge("answer_qa", END)
        builder.add_edge("retrieve_schema", "build_semantic_plan")
        builder.add_conditional_edges(
            "build_semantic_plan",
            self._after_semantic_plan,
            {
                "human_clarification": "human_clarification",
                "prepare_single_database": "prepare_single_database",
                "run_multi_database": "run_multi_database",
            },
        )
        builder.add_edge("human_clarification", "retrieve_schema")
        builder.add_conditional_edges(
            "prepare_single_database",
            lambda state: "human_clarification" if state.get("clarification") else "execute_single_database",
            {
                "human_clarification": "human_clarification",
                "execute_single_database": "execute_single_database",
            },
        )
        builder.add_edge("execute_single_database", END)
        builder.add_conditional_edges(
            "run_multi_database",
            lambda state: "human_clarification" if state.get("clarification") else "end",
            {"human_clarification": "human_clarification", "end": END},
        )
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def run_config(task_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": task_id}}

    def invoke(self, payload: QueryState | Command, task_id: str) -> QueryResult:
        state = self.graph.invoke(payload, config=self.run_config(task_id))
        return self._state_result(state, task_id)

    def _state_result(self, state: dict[str, Any], task_id: str) -> QueryResult:
        if state.get("result"):
            return QueryResult.model_validate(state["result"])
        state = {**state, "task_id": state.get("task_id") or task_id}
        return ResultBuilder.waiting(state)

    def _preprocess(self, state: QueryState) -> dict[str, Any]:
        """生成路由、独立查询和 Schema 检索参数。"""
        decision = self.preprocessor.prepare(state["query"], state.get("route_context", ""))
        execution_log = list(state.get("execution_log") or [])
        if decision.source == "model_unavailable_fallback":
            execution_log.append({
                "stage": "route_fallback",
                "success": True,
                "source": decision.source,
                "action": decision.action,
                "reason": decision.reason,
            })
        return {
            "intent": {
                "action": decision.action,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "response_type": decision.response_type,
                "source": decision.source,
            },
            "direct_response": decision.response,
            "standalone_query": decision.standalone_query,
            "rewritten": decision.rewritten,
            "extraction": decision.retrieval.public(),
            "execution_log": execution_log,
        }

    def _respond_directly(self, state: QueryState) -> dict[str, Any]:
        """返回预处理模型生成的普通回答或自然语言澄清。"""
        result = ResultBuilder.direct_response(
            state["task_id"],
            state.get("direct_response", ""),
            state["intent"],
            list(state.get("execution_log") or []),
        )
        return {
            "workflow_mode": result.workflow_mode,
            "result": result.model_dump(mode="json"),
        }

    def _answer_qa(self, state: QueryState) -> dict[str, Any]:
        qa_result = self.data_qa_agent.run(
            state["query"],
            {
                "short_term": state.get("short_term_context", ""),
                "recent_result": state.get("recent_result_context", ""),
                "selected_tables": state.get("analysis_context", ""),
            },
            state.get("access_scope") or {},
        )
        result = ResultBuilder.qa(
            state["task_id"], qa_result, state["intent"], state.get("analysis_sources") or []
        )
        return {
            "workflow_mode": "qa_report" if qa_result.report else "qa",
            "result": result.model_dump(mode="json"),
        }

    def _retrieve_schema(self, state: QueryState) -> dict[str, Any]:
        standalone_query = state["standalone_query"]
        extraction = state.get("extraction") or {}
        retrieval_terms = [
            *list(extraction.get("retrieval_terms") or []),
            *self._semantic_memory_terms(state.get("semantic_memories") or []),
        ]
        retrieval = self.schema_index.retrieve(
            standalone_query,
            retrieval_terms=retrieval_terms,
            access_scope=state.get("access_scope"),
        )
        workspace = state.get("workspace") or {}
        query_workspace = {
            "schema_fields": list(workspace.get("schema_fields") or []),
            "confirmed_schema_tables": list(workspace.get("confirmed_schema_tables") or []),
            "confirmed_parameters": dict(workspace.get("confirmed_parameters") or {}),
        }
        retrieval = self.schema_index.include_workspace(
            retrieval,
            query_workspace,
            state.get("access_scope"),
        )
        retrieval["extraction"] = extraction
        schema_graph = self.graph_builder.build(
            retrieval["hits"],
            state.get("access_scope"),
        )
        retrieval["schema_graph"] = schema_graph
        databases = sorted({
            str(table.get("database") or schema_graph.get("database") or "short_video_ops")
            for table in schema_graph.get("tables", [])
        })
        return {
            "standalone_query": standalone_query,
            "extraction": extraction,
            "retrieval": retrieval,
            "schema_graph": schema_graph,
            "schema_context": self.graph_builder.context_text(schema_graph),
            "database_names": databases,
            "clarification": None,
            "direct_sql": "",
        }

    @staticmethod
    def _semantic_memory_terms(memories: list[dict[str, Any]]) -> list[str]:
        terms: list[str] = []
        for memory in memories:
            for key in ("name", "value", "evidence"):
                value = memory.get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    terms.append(value)
                else:
                    terms.append(str(value))
        return [term for term in terms if term.strip()]

    def _build_semantic_plan(self, state: QueryState) -> dict[str, Any]:
        """将自然语言意图落到可检查的业务语义计划。"""
        workspace = dict(state.get("workspace") or {})
        memory_context = {
            "confirmed_parameters": dict(workspace.get("confirmed_parameters") or {}),
            "semantic_memories": list(state.get("semantic_memories") or []),
        }
        matches = self.business_semantics.match(state["standalone_query"], memory_context)
        plan = self.semantic_planner.build_plan(state["standalone_query"], memory_context)
        plan_dict = plan.to_dict()
        trace = list(state.get("correction_trace") or [])
        trace.append({
            "stage": "semantic_plan",
            "success": not bool(plan.ambiguities),
            "metric": plan.metric,
            "ambiguity_count": len(plan.ambiguities),
        })
        payload: dict[str, Any] = {
            "semantic_plan": plan_dict,
            "business_semantics": matches.to_dict(),
            "semantic_memories": list(state.get("semantic_memories") or []),
            "semantic_validation": None,
            "correction_trace": trace,
        }
        if plan.ambiguities:
            clarification = self._semantic_clarification(plan)
            if clarification:
                payload.update({
                    "workflow_mode": "semantic_planning_hitl",
                    "clarification": clarification,
                    "direct_sql": "",
                })
        return payload

    @staticmethod
    def _after_semantic_plan(state: QueryState) -> str:
        if state.get("clarification"):
            return "human_clarification"
        return "prepare_single_database" if len(state.get("database_names") or []) <= 1 else "run_multi_database"

    @staticmethod
    def _semantic_clarification(plan: SemanticQueryPlan) -> dict[str, Any] | None:
        if not plan.ambiguities:
            return None
        ambiguity = plan.ambiguities[0]
        options = [
            {
                "id": str(candidate.get("metric") or candidate.get("id") or index),
                "label": str(candidate.get("label") or candidate.get("metric") or candidate),
                "description": str(candidate.get("time_field") or candidate.get("description") or ""),
                "recommended": False,
            }
            for index, candidate in enumerate(ambiguity.candidates)
        ]
        if len(options) < 2:
            return None
        return {
            "parameter": ambiguity.parameter,
            "question": ambiguity.question,
            "reason": ambiguity.reason or "该业务口径会影响 SQL 的指标、时间字段和来源表。",
            "options": options,
        }

    def _human_clarification(self, state: QueryState) -> dict[str, Any]:
        # LangGraph 将 Command(resume=...) 的值作为 interrupt 返回值。
        response = interrupt(state.get("clarification") or {})
        option_id = str(response.get("option_id") if isinstance(response, dict) else response)
        workspace = dict(state.get("workspace") or {})
        payload = state.get("clarification") or {}
        parameter = str(payload.get("parameter") or "other")
        tables = list(workspace.get("confirmed_schema_tables") or [])
        if any(table["id"] == option_id for table in SCHEMA):
            tables = list(dict.fromkeys([*tables, option_id]))
        workspace["confirmed_schema_tables"] = tables
        workspace["confirmed_parameters"] = {
            **dict(workspace.get("confirmed_parameters") or {}),
            parameter: option_id,
        }
        return {"workspace": workspace, "clarification": None, "direct_sql": "", "result": {}}

    def _prepare_single_database(self, state: QueryState) -> dict[str, Any]:
        workspace = dict(state.get("workspace") or {})
        database = (state.get("database_names") or ["short_video_ops"])[0]
        semantic_feedback: dict[str, Any] | None = None
        correction_trace = list(state.get("correction_trace") or [])
        last_decision: dict[str, Any] | None = None
        last_execution: dict[str, Any] | None = None
        last_validation: dict[str, Any] | None = None
        max_attempts = max(1, min(3, self.config.mcp_max_tool_calls))

        for attempt in range(1, max_attempts + 1):
            decision = self.single_database_agent.prepare(
                state["standalone_query"],
                database,
                state["schema_graph"],
                state["schema_context"],
                state["retrieval"],
                workspace,
                state.get("access_scope") or {},
                semantic_plan=state.get("semantic_plan"),
                business_semantics=state.get("business_semantics"),
                semantic_memories=state.get("semantic_memories") or [],
                semantic_feedback=semantic_feedback,
            )
            last_decision = decision
            if decision["action"] == "clarify":
                return {
                    "workflow_mode": "single_database_agent",
                    "clarification": decision["clarification"],
                    "mcp_tool_trace": decision.get("tool_trace", []),
                    "direct_sql": "",
                    "semantic_validation": last_validation,
                    "correction_trace": correction_trace,
                }

            execution = decision["execution"]
            last_execution = execution
            sql = str(execution.get("sql") or "")
            if not bool(execution.get("success")):
                return {
                    "workflow_mode": "single_database_agent",
                    "clarification": None,
                    "mcp_execution": execution,
                    "mcp_tool_trace": decision.get("tool_trace", []),
                    "direct_sql": sql,
                    "sql_source": decision.get("source", "model"),
                    "semantic_validation": last_validation,
                    "correction_trace": correction_trace,
                }

            validation = self._validate_semantic_sql(state, sql)
            last_validation = validation
            if validation.get("valid", True):
                correction_trace.append({
                    "stage": "semantic_validation",
                    "attempt": attempt,
                    "success": True,
                    "issue_count": len(validation.get("issues") or []),
                })
                return {
                    "workflow_mode": "single_database_agent",
                    "clarification": None,
                    "mcp_execution": execution,
                    "mcp_tool_trace": decision.get("tool_trace", []),
                    "direct_sql": sql,
                    "sql_source": decision.get("source", "model"),
                    "semantic_validation": validation,
                    "correction_trace": correction_trace,
                }

            correction = self.correction_router.route(validation.get("issues") or [])
            correction_dict = correction.to_dict()
            correction_trace.append({
                "stage": "semantic_validation",
                "attempt": attempt,
                "success": False,
                "sql": sql,
                "validation": validation,
                "correction": correction_dict,
            })
            if attempt >= max_attempts or correction.action in {"schema_retrieval", "plan_rebuild", "clarification", "give_up"}:
                failed_execution = dict(execution)
                failed_execution["success"] = False
                failed_execution["error"] = self._semantic_error_message(validation, correction_dict)
                return {
                    "workflow_mode": "single_database_agent",
                    "clarification": None,
                    "mcp_execution": failed_execution,
                    "mcp_tool_trace": decision.get("tool_trace", []),
                    "direct_sql": sql,
                    "sql_source": decision.get("source", "model"),
                    "semantic_validation": validation,
                    "correction_trace": correction_trace,
                }
            semantic_feedback = {
                "previous_sql": sql,
                "validation": validation,
                "correction": correction_dict,
                "instruction": "上一条 SQL 可执行但未通过业务语义校验。请只修复校验指出的字段、聚合、过滤、维度或 Join 问题。",
            }

        execution = last_execution or {"sql": "", "success": False, "error": "语义修正未生成SQL"}
        execution = dict(execution)
        execution["success"] = False
        execution["error"] = self._semantic_error_message(
            last_validation or {"issues": []},
            {"action": "give_up", "reason": "语义修正次数已用尽"},
        )
        return {
            "workflow_mode": "single_database_agent",
            "clarification": None,
            "mcp_execution": execution,
            "mcp_tool_trace": (last_decision or {}).get("tool_trace", []),
            "direct_sql": str(execution.get("sql") or ""),
            "sql_source": (last_decision or {}).get("source", "model"),
            "semantic_validation": last_validation,
            "correction_trace": correction_trace,
        }

    def _validate_semantic_sql(self, state: QueryState, sql: str) -> dict[str, Any]:
        raw_plan = state.get("semantic_plan") or {}
        if not raw_plan:
            return {
                "valid": True,
                "skipped": True,
                "reason": "未生成 Semantic Query Plan，跳过语义校验。",
                "issues": [],
            }
        plan = SemanticQueryPlan(**raw_plan)
        rule_result = self.semantic_validator.validate(
            state["standalone_query"],
            plan,
            sql,
            state.get("schema_graph") or {},
        )
        issues = [issue.to_dict() for issue in rule_result.issues]
        judge_result = None
        if rule_result.valid and plan.metric:
            judge = self.semantic_judge.judge(
                state["standalone_query"],
                plan,
                state.get("business_semantics"),
                state.get("schema_graph") or {},
                sql,
                rule_result.ast_summary,
            )
            judge_result = judge.to_dict()
            if not judge.valid:
                issues.extend(issue.to_dict() for issue in judge.issues)
                if not judge.issues:
                    issues.append(
                        SemanticValidationIssue(
                            issue_type="llm_semantic_issue",
                            severity="warning",
                            message=judge.reason or "LLM Judge 认为 SQL 与用户语义不完全一致。",
                        ).to_dict()
                    )
        return {
            "valid": rule_result.valid and (not judge_result or bool(judge_result.get("valid", True))),
            "rule_based": rule_result.to_dict(),
            "llm_judge": judge_result,
            "ast_summary": rule_result.ast_summary,
            "issues": issues,
        }

    @staticmethod
    def _semantic_error_message(
        validation: dict[str, Any],
        correction: dict[str, Any],
    ) -> str:
        issue_text = "；".join(
            str(issue.get("message") or issue.get("issue_type") or issue)
            for issue in validation.get("issues", [])
        )
        reason = str(correction.get("reason") or correction.get("action") or "语义校验未通过")
        return f"业务语义校验未通过：{issue_text or reason}"

    def _execute_single_database(self, state: QueryState) -> dict[str, Any]:
        database = (state.get("database_names") or ["short_video_ops"])[0]
        raw_execution = state.get("mcp_execution") or {}
        execution = SqlExecution(
            sql=str(raw_execution.get("sql") or state.get("direct_sql") or ""),
            success=bool(raw_execution.get("success")),
            columns=list(raw_execution.get("columns") or []),
            rows=list(raw_execution.get("rows") or []),
            error=raw_execution.get("error"),
        )
        trace = list(state.get("mcp_tool_trace") or [])
        log = [
            {
                "stage": "mcp_tool_call",
                "success": not bool(item.get("result", {}).get("error")),
                **item,
            }
            for item in trace
        ]
        if state.get("semantic_validation"):
            log.append({
                "stage": "semantic_validation",
                "success": bool((state.get("semantic_validation") or {}).get("valid", True)),
                "semantic_validation": state.get("semantic_validation"),
                "correction_trace": state.get("correction_trace") or [],
            })
        log.append({
            "stage": "execute_duckdb",
            "success": execution.success,
            "error": execution.error,
            "via": "mcp",
        })
        database_call = next(
            (item for item in reversed(trace) if item.get("tool") == f"query_{database}"),
            {},
        )
        call = {
            "call_index": int(database_call.get("call_index") or 1),
            "database": database,
            "arguments": {
                "mode": "single_database_agent",
                "transport": "mcp_in_process",
                "tool_name": database_call.get("tool"),
                "schema_graph_version": state.get("schema_graph", {}).get("graph_version"),
                "sql_source": state.get("sql_source", "model"),
                "semantic_plan": state.get("semantic_plan"),
            },
            "sql": execution.sql,
            "success": execution.success,
            "row_count": len(execution.rows),
            "error": execution.error,
        }
        if not execution.success:
            result = ResultBuilder.failed(state, execution, log)
            return {"execution_log": log, "tool_calls": [call], "result": result.model_dump(mode="json")}
        try:
            final = self.response_generator.finalize(
                state["standalone_query"], execution, state["schema_context"],
                state.get("analysis_context", ""),
            )
        except PipelineStageError as exc:
            # 结果说明失败时仍保留已成功执行的查询结果。
            log.append({"stage": exc.stage, "success": False, "error": exc.message})
            final = {
                "valid": True,
                "reason": f"{exc.stage}失败",
                "title": "查询结果（文字说明生成失败）",
                "analysis": f"SQL已成功执行，但{exc.stage}失败：{exc.message}",
            }
        result = ResultBuilder.completed(state, [execution], execution, final, [call], log)
        return {"execution_log": log, "tool_calls": [call], "result": result.model_dump(mode="json")}

    def _run_multi_database(self, state: QueryState) -> dict[str, Any]:
        """返回尚未实现的多数据库查询结果。"""
        failure = SqlExecution(
            sql="",
            success=False,
            error="当前仅支持单库直接查询；多库 Handoff 尚未启用。",
        )
        result = ResultBuilder.failed(state, failure, [])
        return {"workflow_mode": "multi_database_pending", "result": result.model_dump(mode="json")}
