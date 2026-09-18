from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from itertools import permutations
import json
import math
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from app.config import BASE_DIR, settings
from app.errors import PipelineStageError
from app.model_client import ModelClient
from app.querying.duckdb_engine import DuckDbEngine
from app.retrieval import SchemaGraphBuilder, SchemaIndex
from app.security import AccessController
from app.semantic import BusinessSemanticLayer, SemanticPlanner, SemanticQueryPlan, SemanticValidator
from app.workflows.query_graph import QueryWorkflow


CASES_PATH = BASE_DIR / "evaluation" / "cases" / "all_cases.jsonl"
RESULT_DIR = BASE_DIR / "evaluation" / "results"


def load_cases(path: Path, scenarios: set[str], classic_only: bool) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if scenarios:
        rows = [row for row in rows if row["scenario"] in scenarios]
    if classic_only:
        rows = [row for row in rows if row["classic"]]
    return rows


def row_values(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [list(row.values()) for row in rows]


def values_equivalent(actual: Any, expected: Any, column: str) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if (
        isinstance(actual, (int, float))
        and isinstance(expected, (int, float))
        and not isinstance(actual, bool)
        and not isinstance(expected, bool)
    ):
        if isinstance(expected, int):
            tolerance = 1e-8
        else:
            normalized = column.lower()
            precision = 4 if any(
                token in normalized
                for token in ("rate", "ratio", "ctr", "retention", "confidence")
            ) else 2
            tolerance = 0.5 * (10 ** -precision) + 1e-9
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=tolerance)

    actual_text = str(actual)
    expected_text = str(expected)
    if actual_text == expected_text:
        return True
    if re.fullmatch(r"\d{4}-\d{2}", expected_text):
        return actual_text.startswith(expected_text + "-01")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected_text):
        return actual_text.startswith(expected_text)
    return False


def rows_equivalent(
    actual_rows: list[list[Any]],
    expected_rows: list[list[Any]],
    expected_columns: list[str],
) -> bool:
    unmatched = list(actual_rows)
    for expected in expected_rows:
        matched_index = next(
            (
                index
                for index, actual in enumerate(unmatched)
                if all(
                    values_equivalent(actual_value, expected_value, column)
                    for actual_value, expected_value, column in zip(
                        actual, expected, expected_columns
                    )
                )
            ),
            None,
        )
        if matched_index is None:
            return False
        unmatched.pop(matched_index)
    return not unmatched


def compare_results(
    actual_columns: list[str],
    actual_rows: list[dict[str, Any]],
    expected_columns: list[str],
    expected_rows: list[dict[str, Any]],
    order_sensitive: bool,
) -> tuple[bool, bool, str]:
    column_match = (
        len(actual_columns) == len(expected_columns)
        and set(actual_columns) == set(expected_columns)
    )
    actual = row_values(actual_rows)
    expected = row_values(expected_rows)
    if len(actual) != len(expected):
        return False, column_match, "返回行数不一致"
    if len(actual_columns) < len(expected_columns):
        return False, column_match, "缺少必要返回列"

    expected_width = len(expected_columns)
    for selected in permutations(range(len(actual_columns)), expected_width):
        projected = [[row[index] for index in selected] for row in actual]
        if rows_equivalent(projected, expected, expected_columns):
            detail = "结果一致"
            if len(actual_columns) != len(expected_columns) or not column_match:
                detail += "（忽略列顺序和辅助列）"
            return True, column_match, detail
    return False, column_match, "返回数据与标准结果不一致"


def field_metrics(retrieved: list[str], expected: list[str]) -> tuple[float, float, int]:
    retrieved_set = set(retrieved)
    expected_set = set(expected)
    true_positive = len(retrieved_set & expected_set)
    recall = true_positive / len(expected_set) if expected_set else 1.0
    precision = true_positive / len(retrieved_set) if retrieved_set else 0.0
    return recall, precision, true_positive


def compare_plan_fields(
    plan: dict[str, Any],
    gold_plan: dict[str, Any] | None,
) -> tuple[float | None, list[str], str]:
    if not gold_plan:
        return None, [], "未标注Gold Plan"
    checked = [
        "metric",
        "aggregation",
        "entity",
        "time_field",
        "time_range",
        "dimensions",
        "filters",
        "source_tables",
    ]
    mismatches: list[str] = []
    total = 0
    matched = 0
    for key in checked:
        if key not in gold_plan:
            continue
        total += 1
        actual = plan.get(key)
        expected = gold_plan.get(key)
        if isinstance(expected, list):
            actual_set = {
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                if isinstance(item, dict) else str(item)
                for item in (actual or [])
            }
            expected_set = {
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                if isinstance(item, dict) else str(item)
                for item in expected
            }
            ok = expected_set.issubset(actual_set)
        else:
            ok = actual == expected
        if ok:
            matched += 1
        else:
            mismatches.append(f"{key}_mismatch")
    if not total:
        return None, [], "Gold Plan未包含可比较字段"
    return matched / total, mismatches, "Gold Plan字段对比完成"


def derived_plan_metric(
    query: str,
    sql: str,
    semantic_layer: BusinessSemanticLayer,
    gold_plan: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any], list[str], str]:
    planner = SemanticPlanner(semantic_layer)
    validator = SemanticValidator(semantic_layer)
    plan = planner.build_plan(query)
    plan_dict = plan.to_dict()
    plan_accuracy, issues, reason = compare_plan_fields(plan_dict, gold_plan)
    if plan_accuracy is not None:
        return plan_accuracy, plan_dict, issues, reason
    if plan.ambiguities or not plan.metric:
        return None, plan_dict, ["plan_unscored"], "Plan存在歧义或未识别指标，跳过近似评分"
    validation = validator.validate(query, SemanticQueryPlan(**plan_dict), sql)
    issue_types = [issue.issue_type for issue in validation.issues]
    return (
        1.0 if validation.valid else 0.0,
        plan_dict,
        issue_types,
        "未标注Gold Plan，使用Gold SQL语义校验近似评分",
    )


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * ratio) - 1))
    return ordered[index]


def summarize(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    count = len(results)
    successful = [row for row in results if row.get("sql_execution_success")]
    correct = [row for row in results if row.get("result_correct")]
    execution_times = [float(row["execution_ms"]) for row in successful]
    recalls = [float(row["schema_recall"]) for row in results if row.get("schema_recall") is not None]
    precisions = [float(row["schema_precision"]) for row in results if row.get("schema_precision") is not None]
    plan_scores = [float(row["plan_accuracy"]) for row in results if row.get("plan_accuracy") is not None]
    plan_issue_counter: Counter[str] = Counter()
    for row in results:
        plan_issue_counter.update(str(item) for item in row.get("plan_issue_types") or [])
    errors = [str(row.get("benchmark_error") or "") for row in results]
    timeout_count = sum(
        any(token in error.lower() for token in ("timed out", "timeout", "超时"))
        for error in errors
    )
    return {
        "mode": mode,
        "case_count": count,
        "schema_recall": round(statistics.fmean(recalls), 6) if recalls else None,
        "schema_precision": round(statistics.fmean(precisions), 6) if precisions else None,
        "plan_accuracy": round(statistics.fmean(plan_scores), 6) if plan_scores else None,
        "plan_evaluated_count": len(plan_scores),
        "plan_error_stats": dict(plan_issue_counter),
        "sql_execution_success_rate": round(len(successful) / count, 6) if count else 0.0,
        "average_execution_ms": round(statistics.fmean(execution_times), 3) if execution_times else 0.0,
        "p95_execution_ms": round(percentile(execution_times, 0.95), 3),
        "result_accuracy": round(len(correct) / count, 6) if count else 0.0,
        "column_name_match_rate": round(
            sum(bool(row.get("column_match")) for row in results) / count, 6
        ) if count else 0.0,
        "clarification_count": sum(row.get("agent_action") == "clarify" for row in results),
        "error_count": sum(bool(row.get("benchmark_error")) for row in results),
        "timeout_count": timeout_count,
        "connection_error_count": sum("WinError 10054" in error for error in errors),
        "model_request_attempt_count": sum(
            int(row.get("model_request_attempt_count") or 0) for row in results
        ),
        "model_retry_count": sum(int(row.get("model_retry_count") or 0) for row in results),
        "model_timeout_event_count": sum(
            int(row.get("model_timeout_event_count") or 0) for row in results
        ),
        "model_connection_error_event_count": sum(
            int(row.get("model_connection_error_event_count") or 0) for row in results
        ),
        "result_accuracy_given_executed_sql": round(
            len(correct) / len(successful), 6
        ) if successful else 0.0,
    }


def run_gold_case(
    case: dict[str, Any],
    engine: DuckDbEngine,
    scope: Any,
    semantic_layer: BusinessSemanticLayer | None = None,
) -> dict[str, Any]:
    semantic_layer = semantic_layer or BusinessSemanticLayer()
    plan_accuracy, semantic_plan, plan_issue_types, plan_reason = derived_plan_metric(
        case["query"],
        case["gold_sql"],
        semantic_layer,
        case.get("gold_plan"),
    )
    execution = engine.execute("short_video_ops", case["gold_sql"], scope)
    result_correct, column_match, reason = compare_results(
        execution.columns,
        execution.rows,
        case["expected_columns"],
        case["expected_rows"],
        bool(case.get("order_sensitive", True)),
    ) if execution.success else (False, False, execution.error or "SQL执行失败")
    return {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "category": case["category"],
        "query_type": case["query_type"],
        "difficulty": case["difficulty"],
        "classic": case["classic"],
        "query": case["query"],
        "schema_recall": None,
        "schema_precision": None,
        "plan_accuracy": plan_accuracy,
        "semantic_plan": semantic_plan,
        "plan_issue_types": plan_issue_types,
        "plan_comparison_reason": plan_reason,
        "retrieved_fields": [],
        "gold_fields": case["gold_fields"],
        "generated_sql": case["gold_sql"],
        "sql_execution_success": execution.success,
        "execution_ms": execution.execution_ms,
        "result_correct": result_correct,
        "column_match": column_match,
        "comparison_reason": reason,
        "agent_action": "gold_sql",
        "benchmark_error": "",
    }


def run_live_case(
    case: dict[str, Any],
    schema_index: SchemaIndex,
    graph_builder: SchemaGraphBuilder,
    workflow: QueryWorkflow,
    controller: AccessController,
) -> dict[str, Any]:
    scope = controller.resolve(f"demo_{case['scenario']}")
    base = {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "category": case["category"],
        "query_type": case["query_type"],
        "difficulty": case["difficulty"],
        "classic": case["classic"],
        "query": case["query"],
        "gold_fields": case["gold_fields"],
    }
    try:
        plan_accuracy, semantic_plan, plan_issue_types, plan_reason = derived_plan_metric(
            case["query"],
            case["gold_sql"],
            workflow.business_semantics,
            case.get("gold_plan"),
        )
        business_matches = workflow.business_semantics.match(case["query"], {})
        plan_model = SemanticQueryPlan(**semantic_plan)
        retrieval_started = time.perf_counter()
        retrieval = schema_index.retrieve(
            case["query"],
            retrieval_terms=case["retrieval_terms"],
            access_scope=scope,
        )
        retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)
        retrieved_fields = [str(hit["doc_id"]) for hit in retrieval["hits"]]
        recall, precision, true_positive = field_metrics(retrieved_fields, case["gold_fields"])
        graph = graph_builder.build(retrieval["hits"], scope)
        schema_context = graph_builder.context_text(graph)
        if not graph.get("tables"):
            return {
                **base,
                "schema_recall": recall,
                "schema_precision": precision,
                "plan_accuracy": plan_accuracy,
                "semantic_plan": semantic_plan,
                "plan_issue_types": plan_issue_types,
                "plan_comparison_reason": plan_reason,
                "schema_true_positive": true_positive,
                "retrieval_ms": retrieval_ms,
                "retrieved_fields": retrieved_fields,
                "generated_sql": "",
                "sql_execution_success": False,
                "execution_ms": 0.0,
                "result_correct": False,
                "column_match": False,
                "comparison_reason": "Schema图为空",
                "agent_action": "not_called",
                "benchmark_error": "Schema图为空",
            }
        if plan_model.ambiguities:
            return {
                **base,
                "schema_recall": recall,
                "schema_precision": precision,
                "plan_accuracy": plan_accuracy,
                "semantic_plan": semantic_plan,
                "plan_issue_types": plan_issue_types,
                "plan_comparison_reason": plan_reason,
                "schema_true_positive": true_positive,
                "retrieval_ms": retrieval_ms,
                "retrieved_fields": retrieved_fields,
                "generated_sql": "",
                "sql_execution_success": False,
                "execution_ms": 0.0,
                "result_correct": False,
                "column_match": False,
                "comparison_reason": "Semantic Query Plan需要澄清",
                "agent_action": "semantic_clarify",
                "semantic_validation": {},
                "benchmark_error": "",
            }
        decision = workflow.single_database_agent.prepare(
            case["query"],
            "short_video_ops",
            graph,
            schema_context,
            retrieval,
            {},
            scope.public(),
            semantic_plan=semantic_plan,
            business_semantics=business_matches.to_dict(),
            semantic_memories=[],
        )
        if decision["action"] == "clarify":
            return {
                **base,
                "schema_recall": recall,
                "schema_precision": precision,
                "plan_accuracy": plan_accuracy,
                "semantic_plan": semantic_plan,
                "plan_issue_types": plan_issue_types,
                "plan_comparison_reason": plan_reason,
                "schema_true_positive": true_positive,
                "retrieval_ms": retrieval_ms,
                "retrieved_fields": retrieved_fields,
                "generated_sql": "",
                "sql_execution_success": False,
                "execution_ms": 0.0,
                "result_correct": False,
                "column_match": False,
                "comparison_reason": "智能体发起澄清",
                "agent_action": "clarify",
                "benchmark_error": "",
            }
        execution = decision["execution"]
        tool_trace = list(decision.get("tool_trace") or [])
        final_trace = tool_trace[-1] if tool_trace else {}
        success = bool(execution.get("success"))
        semantic_validation = {}
        if success:
            semantic_validation = workflow.semantic_validator.validate(
                case["query"],
                plan_model,
                str(execution.get("sql") or ""),
                graph,
            ).to_dict()
        result_correct, column_match, reason = compare_results(
            list(execution.get("columns") or []),
            list(execution.get("rows") or []),
            case["expected_columns"],
            case["expected_rows"],
            bool(case.get("order_sensitive", True)),
        ) if success else (False, False, str(execution.get("error") or "SQL执行失败"))
        return {
            **base,
            "schema_recall": recall,
            "schema_precision": precision,
            "plan_accuracy": plan_accuracy,
            "semantic_plan": semantic_plan,
            "plan_issue_types": plan_issue_types,
            "plan_comparison_reason": plan_reason,
            "schema_true_positive": true_positive,
            "retrieval_ms": retrieval_ms,
            "retrieved_fields": retrieved_fields,
            "generated_sql": str(execution.get("sql") or ""),
            "sql_execution_success": success,
            "execution_ms": float(execution.get("execution_ms") or 0.0),
            "result_correct": result_correct,
            "column_match": column_match,
            "comparison_reason": reason,
            "agent_action": "executed",
            "query_contract": final_trace.get("query_contract", {}),
            "agent_reason": str(final_trace.get("reason") or ""),
            "semantic_validation": semantic_validation,
            "benchmark_error": "",
        }
    except Exception as exc:  # 单条用例失败不终止整个评测。
        error_stage = exc.stage if isinstance(exc, PipelineStageError) else "benchmark"
        return {
            **base,
            "schema_recall": None,
            "schema_precision": None,
            "retrieved_fields": [],
            "generated_sql": "",
            "sql_execution_success": False,
            "execution_ms": 0.0,
            "result_correct": False,
            "column_match": False,
            "comparison_reason": "评测链路异常",
            "agent_action": "error",
            "error_stage": error_stage,
            "benchmark_error": str(exc),
        }


def save_report(results: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_temp = output.with_suffix(output.suffix + ".tmp")
    json_temp.write_text(
        json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    json_temp.replace(output)

    csv_path = output.with_suffix(".csv")
    csv_temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    columns = [
        "case_id", "scenario", "category", "query_type", "difficulty", "classic", "query",
        "schema_recall", "schema_precision", "plan_accuracy", "plan_issue_types",
        "plan_comparison_reason", "retrieval_ms", "sql_execution_success",
        "execution_ms", "result_correct", "column_match", "agent_action",
        "comparison_reason", "generated_sql", "semantic_plan", "semantic_validation",
        "query_contract", "agent_reason", "error_stage",
        "model_request_attempt_count", "model_retry_count", "model_timeout_event_count",
        "model_connection_error_event_count",
        "benchmark_error",
    ]
    with csv_temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    csv_temp.replace(csv_path)

    sql_rate = float(summary.get("sql_execution_success_rate") or 0.0)
    result_rate = float(summary.get("result_accuracy") or 0.0)
    case_count = int(summary.get("case_count") or 0)
    correct_count = sum(bool(row.get("result_correct")) for row in results)
    required_correct = math.ceil(case_count * 0.8)
    shortfall = max(0, required_correct - correct_count)

    scenario_labels = {
        "growth_ops": "用户增长运营",
        "channel_ops": "渠道投放运营",
        "content_ops": "内容运营",
    }
    scenario_lines = ["| 场景 | 用例数 | SQL执行成功率 | 结果正确率 |", "| --- | ---: | ---: | ---: |"]
    for scenario in ("growth_ops", "channel_ops", "content_ops"):
        rows = [row for row in results if row.get("scenario") == scenario]
        if not rows:
            continue
        sql_success = sum(bool(row.get("sql_execution_success")) for row in rows)
        result_success = sum(bool(row.get("result_correct")) for row in rows)
        scenario_lines.append(
            f"| {scenario_labels[scenario]} | {len(rows)} | "
            f"{sql_success / len(rows):.2%} | {result_success / len(rows):.2%} |"
        )

    difficulty_lines = ["| 难度 | 用例数 | SQL执行成功率 | 结果正确率 |", "| --- | ---: | ---: | ---: |"]
    for difficulty in ("简单", "中等", "复杂"):
        rows = [row for row in results if row.get("difficulty") == difficulty]
        if not rows:
            continue
        sql_success = sum(bool(row.get("sql_execution_success")) for row in rows)
        result_success = sum(bool(row.get("result_correct")) for row in rows)
        difficulty_lines.append(
            f"| {difficulty} | {len(rows)} | {sql_success / len(rows):.2%} | "
            f"{result_success / len(rows):.2%} |"
        )

    failure_groups = Counter(
        (
            scenario_labels.get(str(row.get("scenario")), str(row.get("scenario"))),
            str(row.get("category") or "未分类"),
            str(row.get("query_type") or "未分类"),
        )
        for row in results
        if not row.get("result_correct")
    )
    failure_lines = [
        f"- {scenario} / {category} / {query_type}：{count}条"
        for (scenario, category, query_type), count in failure_groups.most_common(8)
    ]
    plan_error_lines = [
        f"- {issue_type}：{count}条"
        for issue_type, count in (summary.get("plan_error_stats") or {}).items()
    ] or ["- 暂无Plan错误统计"]

    conclusion = (
        "SQL执行成功率和查询结果正确率均达到目标。"
        if sql_rate >= 0.9 and result_rate >= 0.8
        else (
            f"SQL执行成功率达到目标；查询结果正确率未达到80%，还差{shortfall}条正确用例。"
            if sql_rate >= 0.9
            else "当前至少有一项核心指标未达到目标，需要继续分析失败用例。"
        )
    )

    markdown = "\n".join([
        "# 问数链路评测报告",
        "",
        f"- 已完成用例：{case_count}",
        f"- 字段级Schema召回率：{float(summary.get('schema_recall') or 0.0):.2%}",
        f"- 字段级Schema准确率：{float(summary.get('schema_precision') or 0.0):.2%}",
        f"- Semantic Plan准确率：{float(summary.get('plan_accuracy') or 0.0):.2%}（已评测{summary.get('plan_evaluated_count', 0)}条）",
        f"- SQL执行成功率：{sql_rate:.2%}（目标≥90%，{'达标' if sql_rate >= 0.9 else '未达标'}）",
        f"- 查询结果正确率：{result_rate:.2%}（目标≥80%，{'达标' if result_rate >= 0.8 else '未达标'}）",
        f"- SQL平均执行时间：{float(summary.get('average_execution_ms') or 0.0):.2f} ms",
        f"- 最终超时数：{summary.get('timeout_count', 0)}",
        f"- 模型原始超时事件：{summary.get('model_timeout_event_count', 0)}",
        f"- 模型重试次数：{summary.get('model_retry_count', 0)}",
        "",
        "## 分场景结果",
        "",
        *scenario_lines,
        "",
        "## 分难度结果",
        "",
        *difficulty_lines,
        "",
        "## 主要失败类型",
        "",
        *failure_lines,
        "",
        "## Plan错误统计",
        "",
        *plan_error_lines,
        "",
        "## 结论",
        "",
        conclusion,
        "",
    ])
    markdown_path = output.with_suffix(".md")
    markdown_temp = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    markdown_temp.write_text(markdown, encoding="utf-8")
    markdown_temp.replace(markdown_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行短视频运营问数评测")
    parser.add_argument("--mode", choices=("gold", "live"), default="gold")
    parser.add_argument("--scope", choices=("all", "classic"), default="all")
    parser.add_argument("--scenario", action="append", choices=("growth_ops", "channel_ops", "content_ops"))
    parser.add_argument("--case-id", action="append", help="只运行指定用例，可重复传入")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_cases(CASES_PATH, set(args.scenario or []), args.scope == "classic")
    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected_ids]
    if args.limit > 0:
        cases = cases[: args.limit]
    controller = AccessController()
    suffix = f"{args.mode}_{args.scope}"
    output = args.output or RESULT_DIR / f"benchmark_{suffix}.json"
    previous: dict[str, dict[str, Any]] = {}
    if args.resume and output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        previous = {str(row["case_id"]): row for row in payload.get("cases", [])}
    results: list[dict[str, Any]] = list(previous.values())
    pending_cases = [case for case in cases if case["case_id"] not in previous]

    if args.mode == "gold":
        engine = DuckDbEngine()
        semantic_layer = BusinessSemanticLayer()
        for index, case in enumerate(pending_cases, 1):
            result = run_gold_case(
                case,
                engine,
                controller.resolve(f"demo_{case['scenario']}"),
                semantic_layer,
            )
            results.append(result)
            print(f"[{index}/{len(pending_cases)}] {case['case_id']} success={result['sql_execution_success']} correct={result['result_correct']}")
    else:
        if not settings.api_key:
            raise RuntimeError("live评测需要在backend/.env配置LLM_API_KEY")
        thread_state = threading.local()

        def evaluate(case: dict[str, Any]) -> dict[str, Any]:
            if not hasattr(thread_state, "schema_resources"):
                schema_index = SchemaIndex(ModelClient(settings), settings)
                schema_index.ensure_built()
                thread_state.schema_resources = (
                    schema_index,
                    SchemaGraphBuilder(),
                )
            schema_index, graph_builder = thread_state.schema_resources

            # 每条用例隔离接口熔断状态，避免一次超时污染同线程的后续用例。
            model_client = ModelClient(settings)
            schema_index.model_client = model_client
            workflow = QueryWorkflow(model_client, schema_index, settings)
            result = run_live_case(
                case,
                schema_index,
                graph_builder,
                workflow,
                controller,
            )
            result.update(model_client.metrics())
            return result

        workers = max(1, min(args.workers, 8))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate, case): case for case in pending_cases}
            for index, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    # 单条用例异常只记录失败，避免中断完整评测。
                    result = {
                        "case_id": case["case_id"],
                        "scenario": case["scenario"],
                        "category": case["category"],
                        "query_type": case["query_type"],
                        "difficulty": case["difficulty"],
                        "classic": case["classic"],
                        "query": case["query"],
                        "gold_fields": case["gold_fields"],
                        "schema_recall": None,
                        "schema_precision": None,
                        "retrieved_fields": [],
                        "generated_sql": "",
                        "sql_execution_success": False,
                        "execution_ms": 0.0,
                        "result_correct": False,
                        "column_match": False,
                        "comparison_reason": "评测线程异常",
                        "agent_action": "error",
                        "error_stage": "benchmark_worker",
                        "benchmark_error": str(exc),
                    }
                results.append(result)
                print(
                    f"[{index}/{len(pending_cases)}] {case['case_id']} "
                    f"recall={result.get('schema_recall')} precision={result.get('schema_precision')} "
                    f"sql={result['sql_execution_success']} result={result['result_correct']}"
                )
                # 每完成一条就保存断点；使用原子替换避免中断时留下半个JSON。
                order = {item["case_id"]: position for position, item in enumerate(cases)}
                results.sort(key=lambda row: order.get(row["case_id"], len(order)))
                save_report(results, summarize(results, args.mode), output)

    order = {case["case_id"]: position for position, case in enumerate(cases)}
    results.sort(key=lambda row: order.get(row["case_id"], len(order)))
    summary = summarize(results, args.mode)
    save_report(results, summary, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
