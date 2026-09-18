from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import BASE_DIR


class MemoryStore:
    """按用户隔离的长期记忆JSON存储。"""

    SEMANTIC_TYPE = "semantic_memory"

    def __init__(self, path: Path | None = None, limit: int = 20) -> None:
        self.path = path or BASE_DIR / "data" / "saved_memories.json"
        self.limit = limit

    def save_result(
        self,
        task_id: str,
        query: str,
        summary: str,
        title: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        user_id: str = "demo_growth_ops",
    ) -> None:
        memory_id = f"result:{task_id}"
        self._upsert(user_id, {
            "id": memory_id,
            "user_id": user_id,
            "kind": "result_table",
            "task_id": task_id,
            "query": query,
            "summary": summary,
            "title": title,
            "columns": columns,
            "rows": rows,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })

    def save_field(
        self,
        table_id: str,
        name: str,
        label: str,
        field_type: str,
        user_id: str = "demo_growth_ops",
    ) -> None:
        memory_id = f"field:{table_id}.{name}"
        self._upsert(user_id, {
            "id": memory_id,
            "user_id": user_id,
            "kind": "schema_field",
            "table_id": table_id,
            "name": name,
            "label": label,
            "field_type": field_type,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })

    def create_semantic_candidate(
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
        user_id: str = "demo_growth_ops",
        access_scope: Any = None,
        version: int = 1,
    ) -> dict[str, Any]:
        """创建一条待人工确认的长期语义记忆候选。"""
        timestamp = self._timestamp()
        item = {
            "id": f"semantic:{uuid.uuid4().hex[:12]}",
            "type": self.SEMANTIC_TYPE,
            "user_id": user_id,
            "access_scope": self._normalize_access_scope(access_scope),
            "kind": kind,
            "name": name,
            "value": deepcopy(value),
            "scope": deepcopy(scope),
            "source": deepcopy(source),
            "evidence": deepcopy(evidence),
            "confidence": confidence,
            "status": "candidate",
            "version": version,
            "expires_at": expires_at,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        conflict = self._semantic_conflict(item)
        if conflict:
            item["conflict"] = conflict
        self._upsert(user_id, item)
        return deepcopy(item)

    def list_semantic_candidates(
        self,
        user_id: str | None = None,
        *,
        access_scope: Any = None,
        include_all: bool = False,
    ) -> list[dict[str, Any]]:
        return [
            deepcopy(item)
            for item in self._read_all()
            if self._is_semantic(item)
            and item.get("status") == "candidate"
            and self._visible_to(item, user_id, access_scope, include_all=include_all)
            and not self._is_expired(item)
        ]

    def confirm_semantic_candidate(
        self,
        memory_id: str,
        user_id: str | None = None,
        *,
        access_scope: Any = None,
        include_all: bool = False,
    ) -> dict[str, Any] | None:
        return self._transition_semantic_candidate(
            memory_id,
            "confirmed",
            user_id,
            access_scope=access_scope,
            include_all=include_all,
        )

    def reject_semantic_candidate(
        self,
        memory_id: str,
        user_id: str | None = None,
        *,
        access_scope: Any = None,
        include_all: bool = False,
    ) -> dict[str, Any] | None:
        return self._transition_semantic_candidate(
            memory_id,
            "rejected",
            user_id,
            access_scope=access_scope,
            include_all=include_all,
        )

    def list_confirmed_semantic_memories(
        self,
        user_id: str | None = None,
        *,
        access_scope: Any = None,
        include_all: bool = False,
    ) -> list[dict[str, Any]]:
        return [
            deepcopy(item)
            for item in self._read_all()
            if self._is_semantic(item)
            and item.get("status") == "confirmed"
            and self._visible_to(item, user_id, access_scope, include_all=include_all)
            and not self._is_expired(item)
        ]

    def delete(
        self,
        memory_id: str,
        user_id: str | None = None,
        *,
        include_all: bool = False,
    ) -> None:
        items = self._read_all()
        self._write([
            item
            for item in items
            if not (
                item.get("id") == memory_id
                and (include_all or user_id is None or self._owner(item) == user_id)
            )
        ])

    def _upsert(self, user_id: str, item: dict[str, Any]) -> None:
        items = self._read_all()
        owned = [
            current
            for current in items
            if self._owner(current) == user_id and current.get("id") != item["id"]
        ]
        others = [current for current in items if self._owner(current) != user_id]
        self._write([item, *owned[: self.limit - 1], *others])

    def _write(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list(
        self,
        user_id: str | None = None,
        *,
        access_scope: Any = None,
        include_all: bool = False,
    ) -> list[dict[str, Any]]:
        items = self._read_all()
        if include_all or user_id is None:
            return [
                deepcopy(item)
                for item in items
                if self._visible_in_memory_list(item)
            ]
        return [
            deepcopy(item)
            for item in items
            if self._visible_to(item, user_id, access_scope, include_all=False)
            and self._visible_in_memory_list(item)
        ]

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    @staticmethod
    def _owner(item: dict[str, Any]) -> str:
        # 兼容升级前未记录所属用户的长期记忆。
        return str(item.get("user_id") or "demo_growth_ops")

    def _transition_semantic_candidate(
        self,
        memory_id: str,
        status: str,
        user_id: str | None,
        *,
        access_scope: Any,
        include_all: bool,
    ) -> dict[str, Any] | None:
        if status not in {"confirmed", "rejected"}:
            raise ValueError(status)
        items = self._read_all()
        updated: dict[str, Any] | None = None
        for item in items:
            if item.get("id") != memory_id:
                continue
            if (
                not self._is_semantic(item)
                or item.get("status") != "candidate"
                or self._is_expired(item)
                or not self._visible_to(
                    item,
                    user_id,
                    access_scope,
                    include_all=include_all,
                )
            ):
                return None
            item["status"] = status
            item["updated_at"] = self._timestamp()
            if status == "confirmed":
                conflict = self._semantic_conflict(item, items=items)
                if conflict:
                    item["conflict"] = conflict
            updated = deepcopy(item)
            break
        if updated is None:
            return None
        self._write(items)
        return updated

    def _semantic_conflict(
        self,
        item: dict[str, Any],
        *,
        items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        conflicts = [
            current
            for current in (items or self._read_all())
            if self._is_semantic(current)
            and current.get("id") != item.get("id")
            and current.get("status") == "confirmed"
            and self._owner(current) == self._owner(item)
            and self._scope_key(current.get("access_scope"))
            == self._scope_key(item.get("access_scope"))
            and not self._is_expired(current)
            and str(current.get("kind") or "") == str(item.get("kind") or "")
            and str(current.get("name") or "") == str(item.get("name") or "")
            and self._scope_key(current.get("scope")) == self._scope_key(item.get("scope"))
        ]
        if not conflicts:
            return None
        return {
            "status": "conflict",
            "message": "同一scope/name/kind下已存在已确认语义记忆",
            "memory_ids": [str(current["id"]) for current in conflicts],
        }

    def _visible_to(
        self,
        item: dict[str, Any],
        user_id: str | None,
        access_scope: Any,
        *,
        include_all: bool,
    ) -> bool:
        if include_all:
            return True
        if user_id is not None and self._owner(item) != user_id:
            return False
        expected_scope = self._normalize_access_scope(access_scope)
        if expected_scope is None or item.get("access_scope") is None:
            return True
        return self._scope_key(item.get("access_scope")) == self._scope_key(expected_scope)

    def _visible_in_memory_list(self, item: dict[str, Any]) -> bool:
        if not self._is_semantic(item):
            return True
        return item.get("status") == "confirmed" and not self._is_expired(item)

    @classmethod
    def _is_semantic(cls, item: dict[str, Any]) -> bool:
        return item.get("type") == cls.SEMANTIC_TYPE or item.get("kind") == cls.SEMANTIC_TYPE

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @classmethod
    def _is_expired(cls, item: dict[str, Any]) -> bool:
        expires_at = cls._parse_time(item.get("expires_at"))
        if expires_at is None:
            return False
        return expires_at <= datetime.now(timezone.utc)

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _scope_key(cls, value: Any) -> str:
        return json.dumps(
            cls._normalize_access_scope(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _normalize_access_scope(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {
                str(key): cls._normalize_access_scope(value[key])
                for key in sorted(value)
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            normalized = [cls._normalize_access_scope(item) for item in value]
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
