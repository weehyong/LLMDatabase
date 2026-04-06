"""
Actor/Component system for database objects — inspired directly by OpenClaw.

OpenClaw's architecture centred on:
  Actor       — a lightweight, unique game entity (player, enemy, item…)
  Component   — a typed behaviour attached to an actor (Physics, Render, AI…)
  ActorFactory — creates actors from XML blueprints

We map those concepts onto database objects:

  Actor         → DbObject  (Table, Index, View, Sequence, …)
  Component     → DbComponent  (ColumnDef, BTreeIndex, Constraint, Trigger, …)
  ActorFactory  → SchemaFactory  (parses schema descriptions, creates DbObjects)

This design lets us mix-and-match behaviours without deep class hierarchies.
For example, a 'Table' actor might have:
  - ColumnDefComponent  (schema — name, type, nullable)
  - PrimaryKeyComponent (constraint)
  - BTreeIndexComponent (index on the PK column)
  - TriggerComponent    (audit trigger)

Each component is self-contained and communicates via the EventBus (see
core.events) rather than holding direct references to sibling components.
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, TypeVar

C = TypeVar("C", bound="DbComponent")


# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------

class ColType(Enum):
    INTEGER = auto()
    REAL = auto()
    TEXT = auto()
    BLOB = auto()
    BOOLEAN = auto()
    TIMESTAMP = auto()
    JSON = auto()


# ---------------------------------------------------------------------------
# DbComponent — base class for all actor behaviours
# ---------------------------------------------------------------------------

class DbComponent(ABC):
    """Base class for all components that can be attached to a DbObject.

    Sub-classes declare their component_type (a unique string tag) and
    implement on_attach / on_detach lifecycle hooks — the same pattern used
    by OpenClaw's ActorComponent.
    """

    @property
    @abstractmethod
    def component_type(self) -> str: ...

    def on_attach(self, owner: "DbObject") -> None:
        """Called after this component is attached to a DbObject."""

    def on_detach(self) -> None:
        """Called before this component is removed from a DbObject."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Concrete Components
# ---------------------------------------------------------------------------

@dataclass
class ColumnDef:
    """Metadata for a single column."""
    name: str
    col_type: ColType
    nullable: bool = True
    default: Any = None
    primary_key: bool = False


class ColumnSetComponent(DbComponent):
    """Stores the ordered list of column definitions for a table (the schema).

    Corresponds to OpenClaw's 'TransformComponent' — the fundamental, always-
    present component that describes what the object *is*.
    """

    component_type = "ColumnSet"

    def __init__(self, columns: List[ColumnDef]) -> None:
        self._columns: List[ColumnDef] = list(columns)
        self._by_name: Dict[str, ColumnDef] = {c.name: c for c in columns}

    @property
    def columns(self) -> List[ColumnDef]:
        return list(self._columns)

    def get(self, name: str) -> Optional[ColumnDef]:
        return self._by_name.get(name)

    def add_column(self, col: ColumnDef) -> None:
        if col.name in self._by_name:
            raise ValueError(f"Column '{col.name}' already exists")
        self._columns.append(col)
        self._by_name[col.name] = col

    def __repr__(self) -> str:
        return f"ColumnSetComponent({[c.name for c in self._columns]})"


class PrimaryKeyComponent(DbComponent):
    """Declares which columns form the primary key.

    Maps to OpenClaw's 'CollisionComponent' — a constraint that rules how
    the object interacts with the world (here: enforces uniqueness).
    """

    component_type = "PrimaryKey"

    def __init__(self, key_columns: List[str]) -> None:
        self._key_columns = key_columns

    @property
    def key_columns(self) -> List[str]:
        return list(self._key_columns)

    def __repr__(self) -> str:
        return f"PrimaryKeyComponent({self._key_columns})"


class IndexComponent(DbComponent):
    """An index on one or more columns.

    In OpenClaw an enemy had a 'PathfinderComponent' to navigate the level;
    here IndexComponent gives a table fast-lookup capability.
    """

    component_type = "Index"

    def __init__(
        self,
        index_name: str,
        columns: List[str],
        unique: bool = False,
        index_type: str = "btree",
    ) -> None:
        self.index_name = index_name
        self.columns = columns
        self.unique = unique
        self.index_type = index_type
        # In-memory representation (simplified): sorted list of (key, row_id)
        self._entries: List[Tuple[Any, int]] = []

    # ------------------------------------------------------------------
    def insert(self, key: Any, row_id: int) -> None:
        import bisect
        bisect.insort(self._entries, (key, row_id))

    def delete(self, key: Any, row_id: int) -> None:
        try:
            self._entries.remove((key, row_id))
        except ValueError:
            pass

    def lookup(self, key: Any) -> List[int]:
        import bisect
        lo = bisect.bisect_left(self._entries, (key, -1))
        results = []
        while lo < len(self._entries) and self._entries[lo][0] == key:
            results.append(self._entries[lo][1])
            lo += 1
        return results

    def range_scan(self, lo: Any, hi: Any) -> Iterator[Tuple[Any, int]]:
        import bisect
        start = bisect.bisect_left(self._entries, (lo, -1))
        for i in range(start, len(self._entries)):
            k, rid = self._entries[i]
            if k > hi:
                break
            yield k, rid

    def __repr__(self) -> str:
        return (
            f"IndexComponent(name={self.index_name!r}, cols={self.columns}, "
            f"unique={self.unique}, entries={len(self._entries)})"
        )


class ForeignKeyComponent(DbComponent):
    """A foreign-key reference constraint.

    Maps to OpenClaw's 'TriggerComponent' — reacts to events on another actor.
    """

    component_type = "ForeignKey"

    def __init__(
        self,
        fk_name: str,
        columns: List[str],
        ref_table: str,
        ref_columns: List[str],
        on_delete: str = "NO ACTION",
        on_update: str = "NO ACTION",
    ) -> None:
        self.fk_name = fk_name
        self.columns = columns
        self.ref_table = ref_table
        self.ref_columns = ref_columns
        self.on_delete = on_delete
        self.on_update = on_update


class CheckComponent(DbComponent):
    """A CHECK constraint expressed as a Python callable predicate."""

    component_type = "Check"

    def __init__(self, name: str, predicate_sql: str) -> None:
        self.name = name
        self.predicate_sql = predicate_sql

    def evaluate(self, row: Dict[str, Any]) -> bool:
        """Evaluate the predicate against a row dict (simple expression eval)."""
        # A full implementation would parse the SQL expression.
        # For now we expose the hook so that the executor can call it.
        try:
            return bool(eval(self.predicate_sql, {}, row))  # noqa: S307
        except Exception:
            return True  # unknown expressions pass by default


class TriggerComponent(DbComponent):
    """A before/after trigger (INSERT/UPDATE/DELETE).

    Corresponds to OpenClaw's event-reaction components (proximity triggers,
    checkpoints, etc.) that fire when something happens near the actor.
    """

    component_type = "Trigger"

    def __init__(
        self,
        trigger_name: str,
        timing: str,       # BEFORE | AFTER
        events: List[str], # INSERT, UPDATE, DELETE
        action: Any,       # callable(old_row, new_row) -> None
    ) -> None:
        self.trigger_name = trigger_name
        self.timing = timing.upper()
        self.events = [e.upper() for e in events]
        self.action = action

    def fire(self, event: str, old_row: Optional[Dict], new_row: Optional[Dict]) -> None:
        if event.upper() in self.events:
            self.action(old_row, new_row)


# ---------------------------------------------------------------------------
# DbObject — the Actor
# ---------------------------------------------------------------------------

class DbObjectType(Enum):
    TABLE = "table"
    INDEX = "index"
    VIEW = "view"
    SEQUENCE = "sequence"


class DbObject:
    """An actor in the database — a Table, Index, View, or Sequence.

    Inspired by OpenClaw's Actor class: a lightweight container identified by
    a unique ID that delegates all behaviour to attached Components.

    Key differences from a traditional class hierarchy:
      - No deep subclassing (Table doesn't extend Relation doesn't extend …)
      - Behaviour added at runtime by attaching/detaching components
      - Cross-cutting concerns (logging, metrics) added as components, not
        woven into base classes
    """

    def __init__(
        self,
        name: str,
        obj_type: DbObjectType,
        actor_id: Optional[str] = None,
    ) -> None:
        self._id = actor_id or str(uuid.uuid4())[:8]
        self._name = name
        self._type = obj_type
        self._components: Dict[str, DbComponent] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @property
    def actor_id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def obj_type(self) -> DbObjectType:
        return self._type

    # ------------------------------------------------------------------
    # Component management (cf. Actor::AddComponent / GetComponent)
    # ------------------------------------------------------------------

    def add_component(self, component: DbComponent) -> None:
        with self._lock:
            if component.component_type in self._components:
                raise ValueError(
                    f"Component '{component.component_type}' already attached to '{self._name}'"
                )
            self._components[component.component_type] = component
        component.on_attach(self)

    def get_component(self, ctype: str) -> Optional[DbComponent]:
        with self._lock:
            return self._components.get(ctype)

    def get_component_typed(self, cls: Type[C]) -> Optional[C]:
        comp = self.get_component(cls.component_type)  # type: ignore[attr-defined]
        return comp if isinstance(comp, cls) else None

    def remove_component(self, ctype: str) -> None:
        with self._lock:
            comp = self._components.pop(ctype, None)
        if comp:
            comp.on_detach()

    def has_component(self, ctype: str) -> bool:
        with self._lock:
            return ctype in self._components

    def all_components(self) -> List[DbComponent]:
        with self._lock:
            return list(self._components.values())

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        comps = list(self._components.keys())
        return f"DbObject(id={self._id!r}, name={self._name!r}, type={self._type.value}, components={comps})"


# ---------------------------------------------------------------------------
# SchemaFactory — creates DbObjects from declarative descriptions
# (cf. OpenClaw's ActorFactory which parsed XML blueprints)
# ---------------------------------------------------------------------------

class SchemaFactory:
    """Creates and configures DbObjects from a schema dictionary.

    The schema dict mirrors OpenClaw's XML actor blueprint:

    {
        "name": "users",
        "type": "table",
        "columns": [
            {"name": "id",    "type": "INTEGER", "primary_key": True},
            {"name": "email", "type": "TEXT",    "nullable": False},
            {"name": "score", "type": "REAL",    "default": 0.0},
        ],
        "indexes": [
            {"name": "idx_email", "columns": ["email"], "unique": True}
        ],
        "checks": [
            {"name": "chk_score", "expr": "score >= 0"}
        ]
    }
    """

    _COL_TYPE_MAP: Dict[str, ColType] = {
        "INTEGER": ColType.INTEGER, "INT": ColType.INTEGER,
        "REAL": ColType.REAL, "FLOAT": ColType.REAL, "DOUBLE": ColType.REAL,
        "TEXT": ColType.TEXT, "VARCHAR": ColType.TEXT, "CHAR": ColType.TEXT,
        "BLOB": ColType.BLOB, "BYTES": ColType.BLOB,
        "BOOLEAN": ColType.BOOLEAN, "BOOL": ColType.BOOLEAN,
        "TIMESTAMP": ColType.TIMESTAMP, "DATETIME": ColType.TIMESTAMP,
        "JSON": ColType.JSON,
    }

    @classmethod
    def create(cls, schema: Dict[str, Any]) -> DbObject:
        name = schema["name"]
        obj_type = DbObjectType(schema.get("type", "table").lower())
        obj = DbObject(name, obj_type)

        # ColumnSet
        pk_cols: List[str] = []
        col_defs: List[ColumnDef] = []
        for col in schema.get("columns", []):
            col_type = cls._COL_TYPE_MAP.get(col["type"].upper(), ColType.TEXT)
            is_pk = col.get("primary_key", False)
            if is_pk:
                pk_cols.append(col["name"])
            col_defs.append(ColumnDef(
                name=col["name"],
                col_type=col_type,
                nullable=col.get("nullable", not is_pk),
                default=col.get("default"),
                primary_key=is_pk,
            ))
        obj.add_component(ColumnSetComponent(col_defs))

        # Primary key
        if pk_cols:
            obj.add_component(PrimaryKeyComponent(pk_cols))
            obj.add_component(IndexComponent(
                index_name=f"pk_{name}",
                columns=pk_cols,
                unique=True,
                index_type="btree",
            ))

        # Additional indexes
        for idx in schema.get("indexes", []):
            obj.add_component(IndexComponent(
                index_name=idx["name"],
                columns=idx["columns"],
                unique=idx.get("unique", False),
                index_type=idx.get("type", "btree"),
            ))

        # Check constraints
        for chk in schema.get("checks", []):
            obj.add_component(CheckComponent(chk["name"], chk["expr"]))

        # Foreign keys
        for fk in schema.get("foreign_keys", []):
            obj.add_component(ForeignKeyComponent(
                fk_name=fk["name"],
                columns=fk["columns"],
                ref_table=fk["ref_table"],
                ref_columns=fk["ref_columns"],
                on_delete=fk.get("on_delete", "NO ACTION"),
                on_update=fk.get("on_update", "NO ACTION"),
            ))

        return obj
