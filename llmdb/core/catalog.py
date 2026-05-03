"""
Catalog — the system catalog (data dictionary) for LLMDatabase.

In OpenClaw the 'GameLogic' class tracked all live actors and served as the
authoritative registry of game objects.  The Catalog plays the same role for
database objects: it is the single source of truth for which tables, indexes,
and views exist and what their schemas are.

The Catalog also implements the statistics interface used by the Planner to
estimate row counts and choose access paths.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmdb.core.actors import DbObject, DbObjectType, SchemaFactory, ColumnSetComponent, IndexComponent


class Catalog:
    """System catalog: owns all DbObjects and exposes query-planner statistics.

    The catalog is persistent: it serialises table schemas to a JSON file
    (catalog.json) inside the database data directory so that schema
    definitions survive restarts.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._tables: Dict[str, DbObject] = {}
        self._stats: Dict[str, Dict[str, int]] = {}  # table → {row_count, page_count}
        self._catalog_path = self._data_dir / "catalog.json"
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._catalog_path.exists():
            return
        try:
            data = json.loads(self._catalog_path.read_text())
            for schema in data.get("tables", []):
                obj = SchemaFactory.create(schema)
                self._tables[obj.name] = obj
            self._stats = data.get("stats", {})
        except Exception:
            pass  # Corrupt catalog — start fresh

    def _save(self) -> None:
        schemas = []
        for name, obj in self._tables.items():
            col_comp = obj.get_component_typed(ColumnSetComponent)
            columns = []
            if col_comp:
                for col in col_comp.columns:
                    columns.append({
                        "name": col.name,
                        "type": col.col_type.name,
                        "nullable": col.nullable,
                        "primary_key": col.primary_key,
                        "default": col.default,
                    })
            schemas.append({"name": name, "type": obj.obj_type.value, "columns": columns})
        data = {"tables": schemas, "stats": self._stats}
        self._catalog_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def create_table(self, schema: Dict[str, Any]) -> DbObject:
        name = schema["name"]
        with self._lock:
            if name in self._tables:
                raise ValueError(f"Table '{name}' already exists")
            obj = SchemaFactory.create(schema)
            self._tables[name] = obj
            self._stats[name] = {"row_count": 0, "page_count": 1}
            self._save()
        return obj

    def drop_table(self, name: str) -> None:
        with self._lock:
            if name not in self._tables:
                raise KeyError(f"Table '{name}' does not exist")
            del self._tables[name]
            self._stats.pop(name, None)
            self._save()

    def get_table(self, name: str) -> Optional[DbObject]:
        with self._lock:
            return self._tables.get(name)

    def list_tables(self) -> List[str]:
        with self._lock:
            return list(self._tables.keys())

    def table_exists(self, name: str) -> bool:
        with self._lock:
            return name in self._tables

    # ------------------------------------------------------------------
    # Statistics (used by Planner)
    # ------------------------------------------------------------------

    def estimated_row_count(self, table: str) -> int:
        with self._lock:
            return self._stats.get(table, {}).get("row_count", 100)

    def update_stats(self, table: str, row_delta: int = 0) -> None:
        with self._lock:
            s = self._stats.setdefault(table, {"row_count": 0, "page_count": 1})
            s["row_count"] = max(0, s["row_count"] + row_delta)
            self._save()

    # ------------------------------------------------------------------
    # Index awareness (used by Planner)
    # ------------------------------------------------------------------

    def find_index_for_predicate(
        self, table: str, predicate_text: str
    ) -> Optional[Dict[str, Any]]:
        """Try to match the WHERE predicate to an available index column.

        Returns a dict with 'name', 'column', 'value' or None.
        """
        obj = self.get_table(table)
        if obj is None:
            return None

        # Simple column = value pattern
        m = re.search(r"(\w+)\s*=\s*(['\"]?)(.+?)\2(?:\s|$)", predicate_text)
        if not m:
            return None
        col_name = m.group(1)
        value = m.group(3)

        # Check if any index covers this column
        for comp in obj.all_components():
            if isinstance(comp, IndexComponent) and col_name in comp.columns:
                return {
                    "name": comp.index_name,
                    "column": col_name,
                    "value": value,
                }
        return None
