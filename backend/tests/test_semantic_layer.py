from __future__ import annotations

import json
import unittest

from app.semantic import (
    BusinessSemanticLayer,
    CorrectionRouter,
    SemanticPlanner,
    SemanticValidationIssue,
    SemanticValidator,
)


class SemanticLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = BusinessSemanticLayer()
        self.planner = SemanticPlanner(self.layer)
        self.validator = SemanticValidator(self.layer)

    def test_planner_builds_serializable_plan_from_business_semantics(self) -> None:
        plan = self.planner.build_plan("按渠道统计本月注册用户数")

        self.assertEqual(plan.metric, "registered_users")
        self.assertEqual(plan.aggregation, "count_distinct")
        self.assertEqual(plan.entity, "user")
        self.assertEqual(plan.time_field, "user_registrations.registered_at")
        self.assertEqual(plan.time_range, "current_month")
        self.assertIn("acquisition_channels.channel_name", plan.dimensions)
        self.assertIn("user_registrations", plan.source_tables)
        self.assertIn("acquisition_channels", plan.source_tables)
        self.assertFalse(plan.ambiguities)
        json.dumps(plan.to_dict(), ensure_ascii=False)

    def test_planner_returns_ambiguity_for_new_users_without_guessing(self) -> None:
        plan = self.planner.build_plan("本月新增用户有多少")

        self.assertIsNone(plan.metric)
        self.assertEqual(len(plan.ambiguities), 1)
        self.assertEqual(plan.ambiguities[0].term, "新增用户")
        self.assertIn(
            "registered_users",
            {candidate["metric"] for candidate in plan.ambiguities[0].candidates},
        )

    def test_user_memory_can_resolve_ambiguous_metric(self) -> None:
        plan = self.planner.build_plan(
            "按渠道统计本月新增用户",
            user_memory={"新增用户默认": "按注册时间统计"},
        )

        self.assertEqual(plan.metric, "registered_users")
        self.assertFalse(plan.ambiguities)
        self.assertEqual(plan.time_field, "user_registrations.registered_at")

    def test_business_layer_matches_query_and_memory_hits(self) -> None:
        hits = self.layer.match(
            "按渠道看本月注册用户",
            user_memory={"常用维度": "平台"},
        )

        self.assertIn("registered_users", {hit.key for hit in hits.metrics})
        self.assertIn("acquisition_channels.channel_name", {hit.key for hit in hits.field_aliases})
        self.assertIn("user_registrations.platform", {hit.key for hit in hits.field_aliases})
        self.assertTrue(hits.rules)

    def test_validator_flags_wrong_time_field(self) -> None:
        plan = self.planner.build_plan("本月注册用户数")
        sql = (
            "SELECT COUNT(DISTINCT user_id) AS user_count "
            "FROM user_registrations "
            "WHERE activated_at >= DATE '2026-09-01'"
        )

        result = self.validator.validate("本月注册用户数", plan, sql)

        self.assertFalse(result.valid)
        self.assertIn("wrong_time_field", {issue.issue_type for issue in result.issues})

    def test_validator_flags_missing_metric_and_dimension_fields(self) -> None:
        plan = self.planner.build_plan("按渠道统计本月注册用户数")
        sql = (
            "SELECT channel_id "
            "FROM user_registrations "
            "WHERE registered_at >= DATE '2026-09-01' "
            "GROUP BY channel_id"
        )

        result = self.validator.validate("按渠道统计本月注册用户数", plan, sql)
        issue_types = {issue.issue_type for issue in result.issues}

        self.assertFalse(result.valid)
        self.assertIn("missing_metric_field", issue_types)
        self.assertIn("missing_dimension", issue_types)

    def test_validator_accepts_structured_parse_failure(self) -> None:
        plan = self.planner.build_plan("本月注册用户数")

        result = self.validator.validate("本月注册用户数", plan, "SELECT FROM")

        self.assertFalse(result.valid)
        self.assertEqual(result.issues[0].issue_type, "sql_parse_error")

    def test_correction_router_routes_error_types(self) -> None:
        router = CorrectionRouter()

        self.assertEqual(
            router.route(SemanticValidationIssue(issue_type="wrong_time_field", message="bad")).action,
            "semantic_repair",
        )
        self.assertEqual(
            router.route(SemanticValidationIssue(issue_type="ambiguous_metric", message="bad")).action,
            "clarification",
        )
        self.assertEqual(
            router.route(SemanticValidationIssue(issue_type="sql_parse_error", message="bad")).action,
            "sql_repair",
        )
        self.assertEqual(
            router.route(SemanticValidationIssue(issue_type="missing_source_table", message="bad")).action,
            "schema_retrieval",
        )


if __name__ == "__main__":
    unittest.main()
