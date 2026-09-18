from __future__ import annotations

import json
from typing import Any

from .models import (
    BusinessSemanticMatches,
    SemanticJudgeResult,
    SemanticQueryPlan,
    SemanticValidationIssue,
)


class SemanticLLMJudge:
    """Thin optional LLM judge. It fails open when no model is available."""

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def judge(
        self,
        query: str,
        plan: SemanticQueryPlan,
        business_hits: BusinessSemanticMatches | dict[str, Any] | None,
        schema: dict[str, Any] | None,
        sql: str,
        ast_summary: dict[str, Any] | None = None,
    ) -> SemanticJudgeResult:
        if self.model is None or getattr(self.model, "enabled", True) is False:
            return SemanticJudgeResult(
                valid=True,
                confidence=0.0,
                skipped=True,
                reason="LLM judge skipped: model unavailable.",
            )

        payload = {
            "query": query,
            "plan": self._serialize(plan),
            "business_hits": self._serialize(business_hits),
            "schema": schema or {},
            "sql": sql,
            "ast_summary": ast_summary or {},
        }
        system = (
            "You are a semantic SQL judge for a business Text2SQL system. "
            "Return JSON with valid(boolean), confidence(number), reason(string), "
            "and optional issues(list of {issue_type,severity,message})."
        )
        user = json.dumps(payload, ensure_ascii=False)

        try:
            raw = self.model.chat_json(system, user)
        except Exception as exc:
            return SemanticJudgeResult(
                valid=True,
                confidence=0.0,
                skipped=True,
                reason=f"LLM judge skipped: {exc}",
            )

        issues = [
            SemanticValidationIssue(
                issue_type=str(item.get("issue_type", "llm_semantic_issue")),
                severity=item.get("severity", "warning"),
                message=str(item.get("message", "")),
                expected=item.get("expected"),
                actual=item.get("actual"),
                metadata=item.get("metadata", {}),
            )
            for item in raw.get("issues", [])
            if isinstance(item, dict)
        ]
        return SemanticJudgeResult(
            valid=bool(raw.get("valid", True)),
            confidence=float(raw.get("confidence", 0.0)),
            reason=str(raw.get("reason", "")),
            skipped=False,
            issues=issues,
            raw_response=raw,
        )

    @staticmethod
    def _serialize(value: Any) -> Any:
        if value is None:
            return None
        dumper = getattr(value, "model_dump", None)
        if callable(dumper):
            return dumper()
        if hasattr(value, "dict"):
            return value.dict()
        if isinstance(value, list):
            return [SemanticLLMJudge._serialize(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): SemanticLLMJudge._serialize(item)
                for key, item in value.items()
            }
        return value
