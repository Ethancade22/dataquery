import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.memory_store import MemoryStore
from app.services.askdata_service import AskDataService


class MemoryStoreTest(unittest.TestCase):
    def test_result_and_field_memories_persist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            store = MemoryStore(path=path, limit=20)

            store.save_result(
                "task-1",
                "查询销售额",
                "查询完成",
                "销售额结果",
                ["地区", "销售额"],
                [{"地区": "华东", "销售额": 100}],
            )
            store.save_field("orders_current", "paid_amount", "实付金额", "数值")

            reloaded = MemoryStore(path=path, limit=20).list()

            self.assertEqual(reloaded[0]["kind"], "schema_field")
            self.assertEqual(reloaded[1]["kind"], "result_table")
            self.assertEqual(reloaded[1]["rows"][0]["销售额"], 100)

    def test_memory_can_be_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            store = MemoryStore(path=path, limit=20)
            store.save_field("customers", "customer_level", "客户等级", "文本")

            store.delete("field:customers.customer_level")

            self.assertEqual(store.list(), [])

    def test_memories_are_persistent_and_isolated_by_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            store = MemoryStore(path=path, limit=20)
            store.save_field(
                "customers", "customer_level", "客户等级", "文本", "user-a"
            )
            store.save_field(
                "orders_current", "paid_amount", "实付金额", "数值", "user-b"
            )

            reloaded = MemoryStore(path=path, limit=20)

            self.assertEqual(len(reloaded.list("user-a")), 1)
            self.assertEqual(reloaded.list("user-a")[0]["name"], "customer_level")
            self.assertEqual(reloaded.list("user-b")[0]["name"], "paid_amount")

    def test_semantic_candidate_can_be_confirmed_or_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            store = MemoryStore(path=path, limit=20)
            access_scope = {
                "roles": ["growth_ops"],
                "allowed_databases": ["short_video_ops"],
                "allowed_tables": ["orders_current"],
            }

            candidate = store.create_semantic_candidate(
                kind="metric_alias",
                name="gmv",
                value={"field": "paid_amount"},
                scope={"table": "orders_current"},
                source={"task_id": "task-1"},
                evidence=[{"text": "GMV等同实付金额"}],
                confidence=0.9,
                user_id="user-a",
                access_scope=access_scope,
            )

            self.assertEqual(candidate["type"], "semantic_memory")
            self.assertEqual(candidate["status"], "candidate")
            self.assertEqual(
                [
                    item["id"]
                    for item in store.list_semantic_candidates(
                        "user-a",
                        access_scope=access_scope,
                    )
                ],
                [candidate["id"]],
            )

            confirmed = store.confirm_semantic_candidate(
                candidate["id"],
                "user-a",
                access_scope=access_scope,
            )
            self.assertIsNotNone(confirmed)
            self.assertEqual(confirmed["status"], "confirmed")
            self.assertEqual(
                store.list_semantic_candidates("user-a", access_scope=access_scope),
                [],
            )
            self.assertEqual(
                [
                    item["id"]
                    for item in store.list_confirmed_semantic_memories(
                        "user-a",
                        access_scope=access_scope,
                    )
                ],
                [candidate["id"]],
            )
            self.assertEqual(
                [item["id"] for item in store.list("user-a", access_scope=access_scope)],
                [candidate["id"]],
            )

            conflict = store.create_semantic_candidate(
                kind="metric_alias",
                name="gmv",
                value={"field": "gmv"},
                scope={"table": "orders_current"},
                user_id="user-a",
                access_scope=access_scope,
            )
            self.assertEqual(conflict["conflict"]["memory_ids"], [candidate["id"]])

            rejected_candidate = store.create_semantic_candidate(
                kind="business_rule",
                name="bad_rule",
                value="ignore",
                scope="global",
                user_id="user-a",
                access_scope=access_scope,
            )
            rejected = store.reject_semantic_candidate(
                rejected_candidate["id"],
                "user-a",
                access_scope=access_scope,
            )
            self.assertIsNotNone(rejected)
            self.assertEqual(rejected["status"], "rejected")
            self.assertNotIn(
                rejected_candidate["id"],
                [
                    item["id"]
                    for item in store.list_semantic_candidates(
                        "user-a",
                        access_scope=access_scope,
                    )
                ],
            )

    def test_semantic_candidates_filter_expired_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            store = MemoryStore(path=path, limit=20)
            access_scope = {"allowed_tables": ["orders_current"]}
            past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

            expired = store.create_semantic_candidate(
                kind="metric_alias",
                name="expired_alias",
                value="paid_amount",
                scope="global",
                expires_at=past,
                user_id="user-a",
                access_scope=access_scope,
            )
            active = store.create_semantic_candidate(
                kind="metric_alias",
                name="active_alias",
                value="paid_amount",
                scope="global",
                expires_at=future,
                user_id="user-a",
                access_scope=access_scope,
            )

            self.assertEqual(
                [
                    item["id"]
                    for item in store.list_semantic_candidates(
                        "user-a",
                        access_scope=access_scope,
                    )
                ],
                [active["id"]],
            )
            self.assertIsNone(
                store.confirm_semantic_candidate(
                    expired["id"],
                    "user-a",
                    access_scope=access_scope,
                )
            )

    def test_semantic_memories_are_isolated_by_user_and_access_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            store = MemoryStore(path=path, limit=20)
            scope_a = {"allowed_tables": ["orders_current"]}
            scope_b = {"allowed_tables": ["customers"]}

            user_a_scope_a = store.create_semantic_candidate(
                kind="metric_alias",
                name="gmv",
                value="paid_amount",
                scope="global",
                user_id="user-a",
                access_scope=scope_a,
            )
            user_b_scope_a = store.create_semantic_candidate(
                kind="metric_alias",
                name="gmv",
                value="paid_amount",
                scope="global",
                user_id="user-b",
                access_scope=scope_a,
            )
            user_a_scope_b = store.create_semantic_candidate(
                kind="metric_alias",
                name="customer_level",
                value="vip",
                scope="global",
                user_id="user-a",
                access_scope=scope_b,
            )

            self.assertEqual(
                [
                    item["id"]
                    for item in store.list_semantic_candidates(
                        "user-a",
                        access_scope=scope_a,
                    )
                ],
                [user_a_scope_a["id"]],
            )
            self.assertEqual(
                [
                    item["id"]
                    for item in store.list_semantic_candidates(
                        "user-b",
                        access_scope=scope_a,
                    )
                ],
                [user_b_scope_a["id"]],
            )
            self.assertEqual(
                [
                    item["id"]
                    for item in store.list_semantic_candidates(
                        "user-a",
                        access_scope=scope_b,
                    )
                ],
                [user_a_scope_b["id"]],
            )

    def test_explicit_semantic_candidate_extraction(self) -> None:
        candidate = AskDataService._extract_explicit_semantic_candidate(
            "以后成交额默认指 paid_amount"
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["kind"], "metric_alias")
        self.assertEqual(candidate["name"], "成交额")
        self.assertEqual(candidate["value"], "paid_amount")
        self.assertIsNone(
            AskDataService._extract_explicit_semantic_candidate("查询本月销售额")
        )


if __name__ == "__main__":
    unittest.main()
