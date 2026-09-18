from __future__ import annotations

import re
from typing import Any

from .knowledge import BusinessSemanticLayer
from .models import JoinPath, SemanticAmbiguity, SemanticFilter, SemanticQueryPlan


class SemanticPlanner:
    def __init__(self, semantic_layer: BusinessSemanticLayer | None = None) -> None:
        self.semantic_layer = semantic_layer or BusinessSemanticLayer()

    def build_plan(self, query: str, user_memory: Any = None) -> SemanticQueryPlan:
        matches = self.semantic_layer.match(query, user_memory)
        resolved_metric_id = self._resolved_metric_from_memory(query, user_memory)
        dimensions = self._dimensions(matches.field_aliases)
        time_range = self._time_range(query)
        ambiguities = [
            SemanticAmbiguity(
                parameter=hit.metadata.get("parameter", "metric"),
                term=hit.metadata.get("term", hit.key),
                candidates=hit.metadata.get("candidates", []),
                question=hit.metadata.get("question", hit.label),
                reason=hit.metadata.get("reason", ""),
            )
            for hit in matches.ambiguities
        ]

        if ambiguities:
            source_tables = self._tables_from_fields(dimensions)
            return SemanticQueryPlan(
                time_range=time_range,
                dimensions=dimensions,
                source_tables=source_tables,
                join_paths=self.semantic_layer.join_paths_for_tables(source_tables),
                ambiguities=ambiguities,
            )

        metric = self.semantic_layer.get_metric(resolved_metric_id) or self._select_metric(matches.metrics)
        if not metric:
            return SemanticQueryPlan(
                time_range=time_range,
                dimensions=dimensions,
                source_tables=self._tables_from_fields(dimensions),
            )

        source_tables = self._unique(list(metric.get("source_tables", [])))
        filters = [
            SemanticFilter(**item)
            for item in metric.get("filters", [])
        ]

        for dimension in dimensions:
            table = self._table_from_field(dimension)
            if table and table not in source_tables:
                source_tables.append(table)

        join_paths = [JoinPath(**item) for item in metric.get("join_paths", [])]
        join_paths.extend(self.semantic_layer.join_paths_for_tables(source_tables))
        join_paths = self._unique_join_paths(join_paths)

        return SemanticQueryPlan(
            metric=metric["id"],
            aggregation=metric.get("aggregation"),
            entity=metric.get("entity"),
            time_field=metric.get("time_field"),
            time_range=time_range,
            dimensions=dimensions,
            filters=filters,
            source_tables=source_tables,
            join_paths=join_paths,
            ambiguities=[],
        )

    def _select_metric(self, metric_hits: list[Any]) -> dict[str, Any] | None:
        if not metric_hits:
            return None
        return dict(metric_hits[0].metadata)

    def _resolved_metric_from_memory(self, query: str, user_memory: Any = None) -> str | None:
        query_text = query.lower()
        for ambiguity in self.semantic_layer.ambiguities:
            term = str(ambiguity.get("term", "")).lower()
            if term and term in query_text:
                return self.semantic_layer.metric_for_ambiguity(ambiguity, user_memory)
        return None

    @staticmethod
    def _dimensions(field_hits: list[Any]) -> list[str]:
        dimensions: list[str] = []
        for hit in field_hits:
            if hit.metadata.get("role") == "dimension":
                dimensions.append(hit.metadata["canonical"])
        return SemanticPlanner._unique(dimensions)

    @staticmethod
    def _time_range(query: str) -> str | None:
        text = query.lower()
        patterns = [
            ("today", ["今天", "今日", "today"]),
            ("yesterday", ["昨天", "昨日", "yesterday"]),
            ("last_7_days", ["近7天", "最近7天", "过去7天", "7天内", "近七天"]),
            ("current_month", ["本月", "这个月", "当月", "current month"]),
            ("last_month", ["上月", "上个月", "last month"]),
            ("current_year", ["今年", "本年", "current year"]),
        ]
        for value, terms in patterns:
            if any(term in text for term in terms):
                return value
        date_match = re.search(r"\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?", query)
        if date_match:
            return date_match.group(0)
        return None

    @staticmethod
    def _table_from_field(field: str) -> str | None:
        parts = field.split(".")
        if len(parts) < 2:
            return None
        return parts[-2]

    @staticmethod
    def _tables_from_fields(fields: list[str]) -> list[str]:
        return SemanticPlanner._unique(
            [
                table
                for field in fields
                if (table := SemanticPlanner._table_from_field(field))
            ]
        )

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    @staticmethod
    def _unique_join_paths(paths: list[JoinPath]) -> list[JoinPath]:
        seen: set[tuple[str, str, str, str]] = set()
        result: list[JoinPath] = []
        for path in paths:
            key = (path.left_table, path.left_field, path.right_table, path.right_field)
            if key in seen:
                continue
            seen.add(key)
            result.append(path)
        return result
