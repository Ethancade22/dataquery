from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


CorrectionAction = Literal[
    "schema_retrieval",
    "plan_rebuild",
    "clarification",
    "semantic_repair",
    "sql_repair",
    "execution_repair",
    "result_review",
    "give_up",
]


class SemanticBaseModel(BaseModel):
    def to_dict(self) -> dict[str, Any]:
        dumper = getattr(self, "model_dump", None)
        if callable(dumper):
            return dumper()
        return self.dict()


class SemanticFilter(SemanticBaseModel):
    field: str
    operator: str = "="
    value: Any = None
    source: str = "business_semantics"


class JoinPath(SemanticBaseModel):
    left_table: str
    left_field: str
    right_table: str
    right_field: str
    relation_type: str = "foreign_key"
    description: str = ""


class SemanticAmbiguity(SemanticBaseModel):
    parameter: str
    term: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    question: str
    reason: str = ""


class SemanticQueryPlan(SemanticBaseModel):
    metric: str | None = None
    aggregation: str | None = None
    entity: str | None = None
    time_field: str | None = None
    time_range: str | None = None
    dimensions: list[str] = Field(default_factory=list)
    filters: list[SemanticFilter] = Field(default_factory=list)
    source_tables: list[str] = Field(default_factory=list)
    join_paths: list[JoinPath] = Field(default_factory=list)
    ambiguities: list[SemanticAmbiguity] = Field(default_factory=list)


class BusinessHit(SemanticBaseModel):
    hit_type: Literal["metric", "field_alias", "rule", "ambiguity", "memory"]
    key: str
    label: str = ""
    score: float = 0.0
    matched_terms: list[str] = Field(default_factory=list)
    source: Literal["query", "memory", "business_semantics"] = "query"
    metadata: dict[str, Any] = Field(default_factory=dict)


class BusinessSemanticMatches(SemanticBaseModel):
    metrics: list[BusinessHit] = Field(default_factory=list)
    field_aliases: list[BusinessHit] = Field(default_factory=list)
    rules: list[BusinessHit] = Field(default_factory=list)
    ambiguities: list[BusinessHit] = Field(default_factory=list)
    memory: list[BusinessHit] = Field(default_factory=list)


class SemanticValidationIssue(SemanticBaseModel):
    issue_type: str
    severity: Literal["info", "warning", "error"] = "error"
    message: str
    expected: Any = None
    actual: Any = None
    location: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticValidationResult(SemanticBaseModel):
    valid: bool
    issues: list[SemanticValidationIssue] = Field(default_factory=list)
    ast_summary: dict[str, Any] = Field(default_factory=dict)
    skipped: bool = False


class SemanticJudgeResult(SemanticBaseModel):
    valid: bool = True
    confidence: float = 0.0
    reason: str = ""
    skipped: bool = False
    issues: list[SemanticValidationIssue] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class CorrectionDecision(SemanticBaseModel):
    action: CorrectionAction
    reason: str
    issue_type: str | None = None
    confidence: float = 1.0
    details: dict[str, Any] = Field(default_factory=dict)
