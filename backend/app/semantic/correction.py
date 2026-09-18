from __future__ import annotations

from typing import Any

from .models import CorrectionDecision, SemanticValidationIssue


class CorrectionRouter:
    def route(
        self,
        issues: list[SemanticValidationIssue | dict[str, Any]] | SemanticValidationIssue | dict[str, Any] | str | None,
    ) -> CorrectionDecision:
        normalized = self._normalize_issues(issues)
        if not normalized:
            return CorrectionDecision(
                action="result_review",
                reason="未发现结构化语义错误，进入结果复核。",
            )

        primary = self._primary_issue(normalized)
        action = self._action_for_issue(primary.issue_type)
        return CorrectionDecision(
            action=action,
            issue_type=primary.issue_type,
            reason=self._reason(action, primary),
            details={"issue": primary.to_dict()},
        )

    @staticmethod
    def _normalize_issues(
        issues: list[SemanticValidationIssue | dict[str, Any]] | SemanticValidationIssue | dict[str, Any] | str | None,
    ) -> list[SemanticValidationIssue]:
        if issues is None:
            return []
        if isinstance(issues, SemanticValidationIssue):
            return [issues]
        if isinstance(issues, str):
            return [
                SemanticValidationIssue(
                    issue_type=issues,
                    message=issues,
                )
            ]
        if isinstance(issues, dict):
            return [SemanticValidationIssue(**issues)]
        return [
            item if isinstance(item, SemanticValidationIssue) else SemanticValidationIssue(**item)
            for item in issues
        ]

    @staticmethod
    def _primary_issue(issues: list[SemanticValidationIssue]) -> SemanticValidationIssue:
        priority = {"error": 0, "warning": 1, "info": 2}
        return sorted(issues, key=lambda item: priority.get(item.severity, 3))[0]

    @staticmethod
    def _action_for_issue(issue_type: str) -> str:
        mapping = {
            "unknown_table": "schema_retrieval",
            "unknown_column": "schema_retrieval",
            "missing_source_table": "schema_retrieval",
            "missing_column": "schema_retrieval",
            "unresolved_ambiguity": "clarification",
            "ambiguous_metric": "clarification",
            "ambiguous_time_field": "clarification",
            "missing_plan": "plan_rebuild",
            "incomplete_plan": "plan_rebuild",
            "plan_conflict": "plan_rebuild",
            "wrong_time_field": "semantic_repair",
            "missing_time_field": "semantic_repair",
            "missing_metric_field": "semantic_repair",
            "missing_dimension": "semantic_repair",
            "missing_filter": "semantic_repair",
            "aggregation_mismatch": "semantic_repair",
            "missing_join_path": "semantic_repair",
            "join_path_mismatch": "semantic_repair",
            "sql_parse_error": "sql_repair",
            "sqlglot_unavailable": "sql_repair",
            "execution_error": "execution_repair",
            "permission_denied": "execution_repair",
            "runtime_error": "execution_repair",
            "result_anomaly": "result_review",
            "low_confidence_result": "result_review",
            "too_many_attempts": "give_up",
        }
        return mapping.get(issue_type, "semantic_repair")

    @staticmethod
    def _reason(action: str, issue: SemanticValidationIssue) -> str:
        reasons = {
            "schema_retrieval": "当前错误更像 Schema 或表字段召回不足，应重新检索相关 Schema。",
            "plan_rebuild": "语义计划不完整或冲突，应重新生成 Semantic Query Plan。",
            "clarification": "用户意图存在关键歧义，应先追问确认业务口径。",
            "semantic_repair": "SQL 可解析，但业务语义与计划不一致，应修复语义字段或口径。",
            "sql_repair": "SQL 结构或解析依赖存在问题，应先修复 SQL。",
            "execution_repair": "错误发生在执行或权限阶段，应进入执行修复。",
            "result_review": "未发现明确可修复错误，进入结果复核。",
            "give_up": "已达到放弃条件，应停止自动重试并交还人工处理。",
        }
        return f"{reasons[action]} 触发 issue: {issue.issue_type}。"
