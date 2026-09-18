from __future__ import annotations

import uuid
import re
from typing import Any

from langgraph.types import Command

from ..config import settings
from ..database import SCHEMA
from ..errors import PipelineStageError
from ..model_client import ModelClient
from ..models import QueryResult
from ..retrieval import SchemaIndex
from ..security import AccessController, AccessScope
from ..workflows.query_graph import QueryWorkflow
from .memory_store import MemoryStore
from .session_context import SessionContext


class AskDataService:
    """查询服务入口，负责会话管理和工作流调用。"""

    def __init__(
        self,
        model_client: ModelClient | None = None,
        schema_index: SchemaIndex | None = None,
    ) -> None:
        self.model_client = model_client or ModelClient(settings)
        self.schema_index = schema_index or SchemaIndex(self.model_client, settings)
        self.config = self.schema_index.config
        self.access_controller = AccessController()
        self.context = SessionContext(self.model_client, self.config)
        self.memories = MemoryStore(limit=self.config.max_saved_memories)
        self.workflow = QueryWorkflow(self.model_client, self.schema_index, self.config)
        self.tasks = self.context.tasks
        self.session_tasks = self.context.session_tasks

    def submit(
        self,
        query: str,
        session_id: str,
        workspace: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> QueryResult:
        normalized = query.strip()
        access_scope = self.access_controller.resolve(user_id)
        scoped_session_id = f"{access_scope.user_id}:{session_id}"
        pending = self.context.latest_pending(scoped_session_id)
        if pending and pending.get("user_id") == access_scope.user_id:
            option_id = self.context.match_clarification(normalized, pending["result"])
            if option_id:
                return self.clarify(
                    pending["result"].task_id,
                    option_id,
                    user_message=normalized,
                    user_id=access_scope.user_id,
                )

        task_id = uuid.uuid4().hex[:12]
        resolved_workspace = self.context.normalize_workspace(workspace)
        self.context.archive.save_message(
            scoped_session_id,
            "user",
            normalized,
            task_id=task_id,
        )
        try:
            result = self.workflow.invoke(
                self._payload(
                    task_id,
                    normalized,
                    scoped_session_id,
                    resolved_workspace,
                    access_scope.public(),
                ),
                task_id,
            )
        except PipelineStageError as exc:
            result = QueryResult(
                task_id=task_id,
                status="failed",
                route="data_qa" if exc.stage == "qa_answer" else "database_query",
                message=f"处理停止：{exc.stage}失败",
                analysis=exc.message,
                route_reason="未使用本地语义兜底",
                steps=[f"{exc.stage}失败，流程已停止"],
                execution_log=[{"stage": exc.stage, "success": False, "error": exc.message}],
                workflow_mode="failed_explicitly",
            )
        self.context.remember(
            scoped_session_id,
            result,
            normalized,
            resolved_workspace,
            access_scope.user_id,
        )
        self._maybe_create_semantic_candidate(normalized, result, access_scope)
        self.context.archive.save_message(
            scoped_session_id,
            "assistant",
            result.analysis or result.message,
            task_id=task_id,
            metadata={"route": result.route, "status": result.status},
        )
        return result

    def clarify(
        self,
        task_id: str,
        option_id: str,
        user_message: str | None = None,
        user_id: str | None = None,
    ) -> QueryResult:
        task = self.tasks.get(task_id)
        if not task:
            raise KeyError(task_id)
        access_scope = self.access_controller.resolve(user_id)
        if task.get("user_id", AccessController.DEFAULT_USER) != access_scope.user_id:
            raise PermissionError("无权访问该查询任务")
        current: QueryResult = task["result"]
        if not current.clarification or option_id not in {item.id for item in current.clarification.options}:
            raise ValueError(option_id)
        option = next(item for item in current.clarification.options if item.id == option_id)
        self.context.archive.save_message(
            task["session_id"],
            "user",
            user_message or option.label,
            task_id=task_id,
            metadata={"type": "clarification", "option_id": option_id},
        )
        result = self.workflow.invoke(Command(resume={"option_id": option_id}), task_id)
        self.context.remember(
            task["session_id"],
            result,
            task["query"],
            task["workspace"],
            access_scope.user_id,
        )
        self.context.archive.save_message(
            task["session_id"],
            "assistant",
            result.analysis or result.message,
            task_id=task_id,
            metadata={"route": result.route, "status": result.status},
        )
        return result

    def search_schema(
        self,
        query: str,
        threshold: float | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        prepared = self.workflow.preprocessor.prepare(query, "")
        extraction = prepared.retrieval
        retrieval = self.schema_index.retrieve(
            query,
            retrieval_terms=extraction.retrieval_terms,
            threshold=threshold,
            access_scope=self.access_controller.resolve(user_id),
        )
        retrieval["extraction"] = extraction.public()
        retrieval["schema_graph"] = self.workflow.graph_builder.build(
            retrieval["hits"],
            self.access_controller.resolve(user_id),
        )
        return retrieval

    def rebuild_schema_index(self) -> dict[str, Any]:
        self.schema_index.ensure_built(force=True)
        return self.schema_index.status()

    def get_task(self, task_id: str, user_id: str | None = None) -> QueryResult:
        if task_id not in self.tasks:
            raise KeyError(task_id)
        scope = self.access_controller.resolve(user_id)
        if self.tasks[task_id].get("user_id", AccessController.DEFAULT_USER) != scope.user_id:
            raise PermissionError("无权访问该查询任务")
        return self.tasks[task_id]["result"]

    def save_memory(self, task_id: str, user_id: str | None = None) -> None:
        task = self.tasks.get(task_id)
        if not task or task["result"].status != "completed":
            raise KeyError(task_id)
        scope = self.access_controller.resolve(user_id)
        if task.get("user_id", AccessController.DEFAULT_USER) != scope.user_id:
            raise PermissionError("无权保存该查询结果")
        result: QueryResult = task["result"]
        self.memories.save_result(
            task_id,
            task["query"],
            result.analysis or result.message,
            result.result_title or "查询结果",
            result.columns,
            result.rows,
            scope.user_id,
        )
        result.saved = True

    def save_field_memory(
        self,
        table_id: str,
        name: str,
        label: str,
        field_type: str,
        user_id: str | None = None,
    ) -> None:
        scope = self.access_controller.resolve(user_id)
        table = next((item for item in SCHEMA if item["id"] == table_id), None)
        database = str(table.get("database") or "short_video_ops") if table else ""
        if not table or not scope.allows_table(database, table_id):
            raise PermissionError("无权保存该字段")
        if not any(field["name"] == name for field in table["fields"]):
            raise ValueError("字段不存在")
        self.memories.save_field(table_id, name, label, field_type, scope.user_id)

    def delete_memory(self, memory_id: str, user_id: str | None = None) -> None:
        scope = self.access_controller.resolve(user_id)
        self.memories.delete(
            memory_id,
            scope.user_id,
            include_all="admin" in scope.roles,
        )

    def list_memories(self, user_id: str | None = None) -> list[dict[str, Any]]:
        scope = self.access_controller.resolve(user_id)
        return self.memories.list(
            scope.user_id,
            access_scope=scope.public(),
            include_all="admin" in scope.roles,
        )

    def create_semantic_memory_candidate(
        self,
        *,
        kind: str,
        name: str,
        value: Any,
        scope: Any,
        source: Any = None,
        evidence: Any = None,
        confidence: float | None = None,
        expires_at: str | None = None,
        user_id: str | None = None,
        version: int = 1,
    ) -> dict[str, Any]:
        access_scope = self.access_controller.resolve(user_id)
        return self.memories.create_semantic_candidate(
            kind=kind,
            name=name,
            value=value,
            scope=scope,
            source=source,
            evidence=evidence,
            confidence=confidence,
            expires_at=expires_at,
            user_id=access_scope.user_id,
            access_scope=access_scope.public(),
            version=version,
        )

    def list_semantic_memory_candidates(
        self,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        scope = self.access_controller.resolve(user_id)
        return self.memories.list_semantic_candidates(
            scope.user_id,
            access_scope=scope.public(),
            include_all="admin" in scope.roles,
        )

    def confirm_semantic_memory_candidate(
        self,
        memory_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        scope = self.access_controller.resolve(user_id)
        item = self.memories.confirm_semantic_candidate(
            memory_id,
            scope.user_id,
            access_scope=scope.public(),
            include_all="admin" in scope.roles,
        )
        if not item:
            raise KeyError(memory_id)
        return item

    def reject_semantic_memory_candidate(
        self,
        memory_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        scope = self.access_controller.resolve(user_id)
        item = self.memories.reject_semantic_candidate(
            memory_id,
            scope.user_id,
            access_scope=scope.public(),
            include_all="admin" in scope.roles,
        )
        if not item:
            raise KeyError(memory_id)
        return item

    def list_confirmed_semantic_memories(
        self,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        scope = self.access_controller.resolve(user_id)
        return self.memories.list_confirmed_semantic_memories(
            scope.user_id,
            access_scope=scope.public(),
            include_all="admin" in scope.roles,
        )

    def list_mcp_tools(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """返回与模型看到的内容一致的MCP工具定义。"""
        scope = self.access_controller.resolve(user_id)
        return self.workflow.mcp_client(scope.public()).list_tools()

    def list_skills(self) -> list[dict[str, Any]]:
        """返回当前启用的应用级Skill配置。"""
        return self.workflow.skills.list()

    def visible_schema(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.access_controller.filter_schema(
            self.access_controller.resolve(user_id)
        )

    def status(self) -> dict[str, Any]:
        return {
            **self.config.public_status(),
            "workflow_engine": "langgraph",
            "human_in_the_loop": "interrupt+checkpointer",
            "mcp": {
                "transport": "in_process",
                "tool_count": len(self.list_mcp_tools()),
            },
            "skills": [item["name"] for item in self.list_skills()],
            "schema_index": self.schema_index.status(),
        }

    def _maybe_create_semantic_candidate(
        self,
        query: str,
        result: QueryResult,
        access_scope: AccessScope,
    ) -> None:
        candidate = self._extract_explicit_semantic_candidate(query)
        if not candidate:
            return
        self.memories.create_semantic_candidate(
            kind=candidate["kind"],
            name=candidate["name"],
            value=candidate["value"],
            scope=candidate["scope"],
            source={"task_id": result.task_id, "route": result.route},
            evidence=[{"role": "user", "text": query}],
            confidence=1.0,
            user_id=access_scope.user_id,
            access_scope=access_scope.public(),
        )

    @staticmethod
    def _extract_explicit_semantic_candidate(query: str) -> dict[str, Any] | None:
        text = query.strip()
        if not any(token in text for token in ("记住", "以后", "后面", "今后")):
            return None
        alias_match = re.search(
            r"(?:记住|以后|后面|今后)?\s*(?:我(?:后面|以后|今后)?说)?[“\"']?"
            r"(?P<name>[^，,。；;：:]{2,30})[”\"']?\s*"
            r"(?:默认)?(?:指|代表|等于|就是|按|默认指|默认是)\s*"
            r"(?P<value>[^，,。；;]{2,80})",
            text,
        )
        if not alias_match:
            return None
        name = alias_match.group("name").strip(" “\"'")
        name = re.sub(r"(?:默认|通常|一般)$", "", name).strip()
        value = alias_match.group("value").strip(" “\"'")
        if not name or not value:
            return None
        return {
            "kind": "metric_alias",
            "name": name,
            "value": value,
            "scope": "user_default",
        }

    def _payload(
        self,
        task_id: str,
        query: str,
        session_id: str,
        workspace: dict[str, Any],
        access_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        analysis_context, analysis_sources = self.context.analysis_context(session_id, workspace)
        scope = AccessScope.from_dict(access_scope) if access_scope else self.access_controller.resolve(None)
        semantic_memories = self.memories.list_confirmed_semantic_memories(
            scope.user_id,
            access_scope=scope.public(),
            include_all="admin" in scope.roles,
        )
        return {
            "task_id": task_id,
            "query": query,
            "session_id": session_id,
            "workspace": self.context.query_workspace(workspace),
            "access_scope": scope.public(),
            "route_context": self.context.route_context(session_id, workspace),
            "short_term_context": self.context.short_term_context(session_id),
            "recent_result_context": self.context.recent_result_context(session_id),
            "analysis_context": analysis_context,
            "analysis_sources": analysis_sources,
            "semantic_memories": semantic_memories,
            "tool_facts": {},
            "execution_log": [],
            "tool_calls": [],
            "clarification": None,
        }

    _normalize_workspace = staticmethod(SessionContext.normalize_workspace)
    _query_workspace = staticmethod(SessionContext.query_workspace)
