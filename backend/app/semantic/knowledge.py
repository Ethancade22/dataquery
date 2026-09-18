from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BusinessHit, BusinessSemanticMatches, JoinPath


SEMANTICS_PATH = Path(__file__).with_name("business_semantics.json")


class BusinessSemanticLayer:
    """Small business-semantic lookup layer backed by a semi-structured JSON file."""

    def __init__(self, path: Path | None = None, data: dict[str, Any] | None = None) -> None:
        self.path = path or SEMANTICS_PATH
        self.data = data or json.loads(self.path.read_text(encoding="utf-8"))
        self.metrics = list(self.data.get("metrics", []))
        self.field_aliases = list(self.data.get("field_aliases", []))
        self.rules = list(self.data.get("rules", []))
        self.ambiguities = list(self.data.get("ambiguities", []))
        self.relations = list(self.data.get("relations", []))
        self._metrics_by_id = {item["id"]: item for item in self.metrics}
        self._rules_by_id = {item["id"]: item for item in self.rules}

    def match(self, query: str, user_memory: Any = None) -> BusinessSemanticMatches:
        query_text = self._normal_text(query)
        memory_text = self._normal_text(user_memory)

        metric_hits = self._match_metrics(query_text, "query")
        memory_metric_hits = self._match_metrics(memory_text, "memory") if memory_text else []
        field_hits = self._match_field_aliases(query_text, "query")
        memory_field_hits = self._match_field_aliases(memory_text, "memory") if memory_text else []
        rule_hits = self._rules_for_metrics(metric_hits + memory_metric_hits)
        ambiguity_hits = self._match_ambiguities(query_text, user_memory)

        return BusinessSemanticMatches(
            metrics=self._dedupe_hits(metric_hits + memory_metric_hits),
            field_aliases=self._dedupe_hits(field_hits + memory_field_hits),
            rules=rule_hits,
            ambiguities=ambiguity_hits,
            memory=[
                hit
                for hit in memory_metric_hits + memory_field_hits
                if hit.source == "memory"
            ],
        )

    def get_metric(self, metric_id: str | None) -> dict[str, Any] | None:
        if not metric_id:
            return None
        return self._metrics_by_id.get(metric_id)

    def get_rule(self, rule_id: str) -> dict[str, Any] | None:
        return self._rules_by_id.get(rule_id)

    def metric_for_ambiguity(self, ambiguity: dict[str, Any], user_memory: Any = None) -> str | None:
        memory_text = self._memory_preference_text(user_memory)
        if not memory_text:
            return None
        ambiguous_term = str(ambiguity.get("term", "")).lower()
        matched: list[str] = []
        for candidate in ambiguity.get("candidates", []):
            metric_id = candidate.get("metric")
            metric = self.get_metric(metric_id)
            if not metric:
                continue
            terms = [
                metric["id"],
                *(metric.get("aliases") or []),
                *(metric.get("disambiguation_terms") or []),
                candidate.get("time_field", ""),
            ]
            terms = [
                term
                for term in terms
                if term and ambiguous_term not in str(term).lower()
            ]
            if self._matching_terms(memory_text, terms):
                matched.append(metric["id"])
        unique = self._unique(matched)
        if len(unique) == 1:
            return unique[0]
        return None

    def join_paths_for_tables(self, tables: list[str]) -> list[JoinPath]:
        table_set = {self._short_table(table) for table in tables}
        paths: list[JoinPath] = []
        for relation in self.relations:
            left = self._short_table(str(relation.get("left_table", "")))
            right = self._short_table(str(relation.get("right_table", "")))
            if left in table_set and right in table_set:
                paths.append(JoinPath(**relation))
        return paths

    def known_time_fields(self) -> set[str]:
        fields = {
            item["canonical"]
            for item in self.field_aliases
            if item.get("role") == "time"
        }
        fields.update(
            metric["time_field"]
            for metric in self.metrics
            if metric.get("time_field")
        )
        return fields

    def _match_metrics(self, text: str, source: str) -> list[BusinessHit]:
        if not text:
            return []
        hits: list[BusinessHit] = []
        for metric in self.metrics:
            terms = [
                metric["id"],
                metric.get("name", ""),
                metric.get("metric_field", ""),
                *(metric.get("aliases") or []),
            ]
            matched = self._matching_terms(text, terms)
            if not matched:
                continue
            score = 3.0 if source == "query" else 1.5
            score += min(max(len(max(matched, key=len)) / 10, 0.1), 1.5)
            hits.append(
                BusinessHit(
                    hit_type="metric",
                    key=metric["id"],
                    label=metric.get("name", metric["id"]),
                    score=score,
                    matched_terms=matched,
                    source=source,  # type: ignore[arg-type]
                    metadata=metric,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)

    def _match_field_aliases(self, text: str, source: str) -> list[BusinessHit]:
        if not text:
            return []
        hits: list[BusinessHit] = []
        for field in self.field_aliases:
            terms = [
                field["canonical"],
                field.get("label", ""),
                *(field.get("aliases") or []),
            ]
            matched = self._matching_terms(text, terms)
            if not matched:
                continue
            hits.append(
                BusinessHit(
                    hit_type="field_alias",
                    key=field["canonical"],
                    label=field.get("label", field["canonical"]),
                    score=2.0 if source == "query" else 1.0,
                    matched_terms=matched,
                    source=source,  # type: ignore[arg-type]
                    metadata=field,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)

    def _rules_for_metrics(self, metric_hits: list[BusinessHit]) -> list[BusinessHit]:
        seen: set[str] = set()
        hits: list[BusinessHit] = []
        for hit in metric_hits:
            metric = hit.metadata
            for rule_id in metric.get("rules", []):
                if rule_id in seen:
                    continue
                rule = self.get_rule(rule_id)
                if not rule:
                    continue
                seen.add(rule_id)
                hits.append(
                    BusinessHit(
                        hit_type="rule",
                        key=rule_id,
                        label=rule.get("label", rule_id),
                        score=hit.score,
                        matched_terms=hit.matched_terms,
                        source="business_semantics",
                        metadata=rule,
                    )
                )
        return hits

    def _match_ambiguities(self, query_text: str, user_memory: Any) -> list[BusinessHit]:
        hits: list[BusinessHit] = []
        for ambiguity in self.ambiguities:
            term = str(ambiguity.get("term", ""))
            if not term or term.lower() not in query_text:
                continue
            if term == "新增用户" and "新增用户数" in query_text:
                continue
            if self.metric_for_ambiguity(ambiguity, user_memory):
                continue
            hits.append(
                BusinessHit(
                    hit_type="ambiguity",
                    key=term,
                    label=ambiguity.get("question", term),
                    score=5.0,
                    matched_terms=[term],
                    source="query",
                    metadata=ambiguity,
                )
            )
        return hits

    @staticmethod
    def _normal_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.lower()
        return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()

    @staticmethod
    def _memory_preference_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, dict):
            return " ".join(
                BusinessSemanticLayer._normal_text(item)
                for item in value.values()
            )
        if isinstance(value, (list, tuple, set)):
            return " ".join(
                BusinessSemanticLayer._normal_text(item)
                for item in value
            )
        return BusinessSemanticLayer._normal_text(value)

    @staticmethod
    def _matching_terms(text: str, terms: list[str]) -> list[str]:
        matches: list[str] = []
        for term in terms:
            if not term:
                continue
            normalized = str(term).lower()
            if normalized and normalized in text:
                matches.append(str(term))
        return BusinessSemanticLayer._unique(matches)

    @staticmethod
    def _dedupe_hits(hits: list[BusinessHit]) -> list[BusinessHit]:
        best: dict[tuple[str, str], BusinessHit] = {}
        for hit in hits:
            key = (hit.hit_type, hit.key)
            if key not in best or hit.score > best[key].score:
                best[key] = hit
        return sorted(best.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _short_table(table: str) -> str:
        return table.split(".")[-1].strip('"').lower()

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result
