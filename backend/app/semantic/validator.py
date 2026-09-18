from __future__ import annotations

from typing import Any

from .knowledge import BusinessSemanticLayer
from .models import (
    SemanticQueryPlan,
    SemanticValidationIssue,
    SemanticValidationResult,
)


class SemanticValidator:
    def __init__(self, semantic_layer: BusinessSemanticLayer | None = None) -> None:
        self.semantic_layer = semantic_layer or BusinessSemanticLayer()

    def validate(
        self,
        query: str,
        plan: SemanticQueryPlan,
        sql: str,
        schema: dict[str, Any] | None = None,
    ) -> SemanticValidationResult:
        ast_result = self._parse_sql(sql)
        if ast_result["issue"]:
            return SemanticValidationResult(
                valid=False,
                issues=[ast_result["issue"]],
                ast_summary=ast_result["summary"],
                skipped=ast_result["skipped"],
            )

        ast_summary = ast_result["summary"]
        issues: list[SemanticValidationIssue] = []
        issues.extend(self._check_source_tables(plan, ast_summary))
        issues.extend(self._check_time_field(plan, ast_summary))
        issues.extend(self._check_metric(plan, sql, ast_summary))
        issues.extend(self._check_dimensions(plan, ast_summary))
        issues.extend(self._check_filters(plan, ast_summary))
        issues.extend(self._check_join_paths(plan, ast_summary))

        return SemanticValidationResult(
            valid=not any(issue.severity == "error" for issue in issues),
            issues=issues,
            ast_summary=ast_summary,
        )

    def _parse_sql(self, sql: str) -> dict[str, Any]:
        try:
            import sqlglot
            from sqlglot import exp
            from sqlglot.errors import ParseError
        except Exception as exc:  # pragma: no cover - exercised only without dependency.
            return {
                "issue": SemanticValidationIssue(
                    issue_type="sqlglot_unavailable",
                    severity="error",
                    message="sqlglot 不可用，无法执行 SQL AST 语义校验。",
                    actual=str(exc),
                ),
                "summary": {"parser": "sqlglot", "error": str(exc)},
                "skipped": True,
            }

        try:
            ast = sqlglot.parse_one(sql, read="duckdb")
        except ParseError as exc:
            return {
                "issue": SemanticValidationIssue(
                    issue_type="sql_parse_error",
                    severity="error",
                    message="SQL 解析失败，无法执行语义校验。",
                    actual=str(exc),
                ),
                "summary": {"parser": "sqlglot", "error": str(exc)},
                "skipped": False,
            }
        except Exception as exc:
            return {
                "issue": SemanticValidationIssue(
                    issue_type="sql_parse_error",
                    severity="error",
                    message="SQL 解析异常，无法执行语义校验。",
                    actual=str(exc),
                ),
                "summary": {"parser": "sqlglot", "error": str(exc)},
                "skipped": False,
            }

        tables = sorted(
            {
                self._normalize_name(table.name)
                for table in ast.find_all(exp.Table)
                if table.name
            }
        )
        columns = []
        for column in ast.find_all(exp.Column):
            columns.append(
                {
                    "name": self._normalize_name(column.name),
                    "table": self._normalize_name(column.table) if column.table else "",
                    "sql": column.sql(dialect="duckdb"),
                }
            )
        aggregate_functions = sorted(
            {
                node.key.upper()
                for node in ast.walk()
                if node.key in {"count", "sum", "avg", "max", "min"}
            }
        )
        group_columns = [
            self._normalize_name(column.name)
            for group in ast.find_all(exp.Group)
            for column in group.find_all(exp.Column)
        ]
        where = next(ast.find_all(exp.Where), None)
        joins = list(ast.find_all(exp.Join))

        return {
            "issue": None,
            "summary": {
                "parser": "sqlglot",
                "tables": tables,
                "columns": columns,
                "column_names": sorted({column["name"] for column in columns}),
                "aggregate_functions": aggregate_functions,
                "group_columns": sorted(set(group_columns)),
                "has_where": where is not None,
                "join_count": len(joins),
            },
            "skipped": False,
        }

    def _check_source_tables(
        self,
        plan: SemanticQueryPlan,
        ast_summary: dict[str, Any],
    ) -> list[SemanticValidationIssue]:
        issues: list[SemanticValidationIssue] = []
        actual_tables = set(ast_summary.get("tables", []))
        for table in plan.source_tables:
            expected = self._normalize_table(table)
            if expected not in actual_tables:
                issues.append(
                    SemanticValidationIssue(
                        issue_type="missing_source_table",
                        message="SQL 缺少语义计划要求的数据表。",
                        expected=expected,
                        actual=sorted(actual_tables),
                    )
                )
        return issues

    def _check_time_field(
        self,
        plan: SemanticQueryPlan,
        ast_summary: dict[str, Any],
    ) -> list[SemanticValidationIssue]:
        if not plan.time_field or not plan.time_range:
            return []
        expected = plan.time_field
        if self._has_field(ast_summary, expected):
            return []

        wrong_fields = sorted(
            field
            for field in self.semantic_layer.known_time_fields()
            if self._has_field(ast_summary, field)
            and self._field_name(field) != self._field_name(expected)
        )
        if wrong_fields:
            return [
                SemanticValidationIssue(
                    issue_type="wrong_time_field",
                    message="SQL 使用了不符合业务口径的时间字段。",
                    expected=expected,
                    actual=wrong_fields,
                )
            ]
        return [
            SemanticValidationIssue(
                issue_type="missing_time_field",
                message="SQL 缺少语义计划要求的时间字段。",
                expected=expected,
                actual=ast_summary.get("column_names", []),
            )
        ]

    def _check_metric(
        self,
        plan: SemanticQueryPlan,
        sql: str,
        ast_summary: dict[str, Any],
    ) -> list[SemanticValidationIssue]:
        issues: list[SemanticValidationIssue] = []
        metric = self.semantic_layer.get_metric(plan.metric)
        if metric and metric.get("metric_field") and not self._has_field(ast_summary, metric["metric_field"]):
            issues.append(
                SemanticValidationIssue(
                    issue_type="missing_metric_field",
                    message="SQL 缺少指标计算需要的字段。",
                    expected=metric["metric_field"],
                    actual=ast_summary.get("column_names", []),
                )
            )
        if plan.aggregation:
            aggregate = self._expected_aggregate(plan.aggregation)
            if aggregate and aggregate not in sql.upper():
                issues.append(
                    SemanticValidationIssue(
                        issue_type="aggregation_mismatch",
                        message="SQL 聚合函数与语义计划不一致。",
                        expected=aggregate,
                        actual=ast_summary.get("aggregate_functions", []),
                    )
                )
        return issues

    def _check_dimensions(
        self,
        plan: SemanticQueryPlan,
        ast_summary: dict[str, Any],
    ) -> list[SemanticValidationIssue]:
        issues: list[SemanticValidationIssue] = []
        for dimension in plan.dimensions:
            if not self._has_field(ast_summary, dimension):
                issues.append(
                    SemanticValidationIssue(
                        issue_type="missing_dimension",
                        message="SQL 缺少语义计划要求的维度字段。",
                        expected=dimension,
                        actual=ast_summary.get("column_names", []),
                    )
                )
        return issues

    def _check_filters(
        self,
        plan: SemanticQueryPlan,
        ast_summary: dict[str, Any],
    ) -> list[SemanticValidationIssue]:
        issues: list[SemanticValidationIssue] = []
        for item in plan.filters:
            if not self._has_field(ast_summary, item.field):
                issues.append(
                    SemanticValidationIssue(
                        issue_type="missing_filter",
                        severity="warning",
                        message="SQL 缺少业务语义建议的过滤字段。",
                        expected=item.to_dict(),
                        actual=ast_summary.get("column_names", []),
                    )
                )
        return issues

    def _check_join_paths(
        self,
        plan: SemanticQueryPlan,
        ast_summary: dict[str, Any],
    ) -> list[SemanticValidationIssue]:
        if len(plan.source_tables) < 2:
            return []
        actual_tables = set(ast_summary.get("tables", []))
        if ast_summary.get("join_count", 0) == 0:
            return [
                SemanticValidationIssue(
                    issue_type="missing_join_path",
                    message="语义计划需要多表关联，但 SQL 未包含 JOIN。",
                    expected=[path.to_dict() for path in plan.join_paths],
                    actual=sorted(actual_tables),
                )
            ]
        return []

    @staticmethod
    def _expected_aggregate(aggregation: str) -> str | None:
        lowered = aggregation.lower()
        if lowered.startswith("count"):
            return "COUNT"
        if lowered == "sum":
            return "SUM"
        if lowered == "avg":
            return "AVG"
        if lowered == "max":
            return "MAX"
        if lowered == "min":
            return "MIN"
        return None

    @staticmethod
    def _has_field(ast_summary: dict[str, Any], field: str) -> bool:
        expected_name = SemanticValidator._field_name(field)
        return expected_name in set(ast_summary.get("column_names", []))

    @staticmethod
    def _field_name(field: str) -> str:
        return SemanticValidator._normalize_name(field.split(".")[-1])

    @staticmethod
    def _normalize_table(table: str) -> str:
        return SemanticValidator._normalize_name(table.split(".")[-1])

    @staticmethod
    def _normalize_name(value: str) -> str:
        return str(value).strip('"').lower()
