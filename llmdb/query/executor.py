"""
Query Executor — walks the plan tree and produces rows.

Inspired by OpenClaw's ProcessManager tick: each call to execute() is like
a game-loop tick that advances the plan tree until results are exhausted.

The executor uses the Volcano iterator model: each plan node is a lazy
generator that pulls tuples from its child.  The executor orchestrates the
top-level iteration and collects the result set.
"""

from __future__ import annotations

import ast
import operator
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

from llmdb.core.actors import (
    ColumnSetComponent, IndexComponent, PrimaryKeyComponent,
    CheckComponent, TriggerComponent,
)
from llmdb.core.catalog import Catalog
from llmdb.query.planner import (
    CreateIndexNode, CreateTableNode, DeleteNode, DropIndexNode, DropTableNode,
    FilterNode, IndexScanNode, InsertNode, LimitNode, NestedLoopJoinNode,
    NoOpNode, PlanNode, ProjectionNode, SeqScanNode, SortNode, TransactionNode,
    UpdateNode,
)

Row = Dict[str, Any]
RowIter = Iterator[Row]


# ---------------------------------------------------------------------------
# Simple row store (in-memory, per-table)
# ---------------------------------------------------------------------------
# A real DBMS would use the Buffer Pool + Storage Manager here.
# We include both an in-memory store (for fast testing) and hooks to the
# buffer/storage layer for persistence.

class _RowStore:
    """Per-table in-memory row storage with auto-increment rowid."""

    def __init__(self) -> None:
        self._rows: Dict[int, Row] = {}
        self._next_rowid = 1

    def insert(self, row: Row) -> int:
        rowid = self._next_rowid
        self._next_rowid += 1
        self._rows[rowid] = dict(row)
        return rowid

    def scan(self) -> Iterator[Tuple[int, Row]]:
        yield from self._rows.items()

    def update(self, rowid: int, updates: Dict[str, Any]) -> None:
        if rowid in self._rows:
            self._rows[rowid].update(updates)

    def delete(self, rowid: int) -> None:
        self._rows.pop(rowid, None)

    def get(self, rowid: int) -> Optional[Row]:
        return self._rows.get(rowid)

    def count(self) -> int:
        return len(self._rows)


# ---------------------------------------------------------------------------
# Expression evaluator (safe subset)
# ---------------------------------------------------------------------------

def _eval_expr(expr_text: str, row: Row) -> Any:
    """Evaluate a simple SQL expression against a row dict.

    Supports: column references, literals, comparison operators, AND/OR,
    LIKE (via regex), IS NULL / IS NOT NULL, arithmetic.
    """
    # Replace SQL operators/keywords with Python equivalents
    text = expr_text.strip()
    text = re.sub(r"\bIS\s+NOT\s+NULL\b", "is not None", text, flags=re.IGNORECASE)
    text = re.sub(r"\bIS\s+NULL\b", "is None", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAND\b", " and ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bOR\b", " or ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNOT\b", " not ", text, flags=re.IGNORECASE)
    # Translate SQL '=' (single equals) to Python '==' — carefully avoid
    # touching '==', '!=', '<=', '>=' which are already Python-compatible.
    text = re.sub(r"(?<![=!<>])=(?!=)", "==", text)

    # LIKE → python regex (simple: % → .*, _ → .)
    def _like_to_regex(m: re.Match) -> str:
        col = m.group(1).strip()
        pattern = m.group(2).strip("'\"")
        py_pattern = re.escape(pattern).replace(r"\%", ".*").replace(r"\_", ".")
        return f'bool(re.match("{py_pattern}", str({col})))'

    text = re.sub(
        r"(\w+)\s+LIKE\s+(['\"].*?['\"])",
        _like_to_regex,
        text,
        flags=re.IGNORECASE,
    )

    # Build evaluation namespace: row columns + safe builtins
    namespace: Dict[str, Any] = {
        **row,
        "re": re,
        "None": None,
        "True": True,
        "False": False,
    }
    try:
        return eval(text, {"__builtins__": {}}, namespace)  # noqa: S307
    except Exception:
        return True  # unknown expressions pass through


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    rows: List[Row]
    row_count: int
    affected_rows: int
    elapsed_ms: float
    error: Optional[str] = None


class Executor:
    """Executes a plan tree against the catalog and row store.

    The executor maintains a per-session row store dict.  In production,
    row data would live in the buffer pool; the in-memory store here provides
    a correct, testable foundation.
    """

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        self._stores: Dict[str, _RowStore] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, plan: PlanNode) -> ExecutionResult:
        t0 = time.monotonic()
        try:
            rows, affected = self._execute_node(plan)
            rows = list(rows)
            elapsed = (time.monotonic() - t0) * 1000
            return ExecutionResult(rows, len(rows), affected, elapsed)
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            return ExecutionResult([], 0, 0, elapsed, error=str(exc))

    # ------------------------------------------------------------------
    # Node dispatcher
    # ------------------------------------------------------------------

    def _execute_node(self, node: PlanNode) -> Tuple[RowIter, int]:
        if isinstance(node, SeqScanNode):
            return self._seq_scan(node), 0
        elif isinstance(node, IndexScanNode):
            return self._index_scan(node), 0
        elif isinstance(node, FilterNode):
            return self._filter(node), 0
        elif isinstance(node, ProjectionNode):
            return self._project(node), 0
        elif isinstance(node, SortNode):
            return self._sort(node), 0
        elif isinstance(node, LimitNode):
            return self._limit(node), 0
        elif isinstance(node, NestedLoopJoinNode):
            return self._nested_loop_join(node), 0
        elif isinstance(node, InsertNode):
            return iter([]), self._insert(node)
        elif isinstance(node, UpdateNode):
            return iter([]), self._update(node)
        elif isinstance(node, DeleteNode):
            return iter([]), self._delete(node)
        elif isinstance(node, CreateTableNode):
            return iter([]), self._create_table(node)
        elif isinstance(node, DropTableNode):
            return iter([]), self._drop_table(node)
        elif isinstance(node, CreateIndexNode):
            return iter([]), self._create_index(node)
        elif isinstance(node, DropIndexNode):
            return iter([]), 0  # stub
        elif isinstance(node, TransactionNode):
            return iter([{"status": node.operation}]), 0
        elif isinstance(node, NoOpNode):
            return iter([]), 0
        raise NotImplementedError(f"Unknown plan node: {type(node).__name__}")

    # ------------------------------------------------------------------
    # Access paths
    # ------------------------------------------------------------------

    def _seq_scan(self, node: SeqScanNode) -> RowIter:
        store = self._get_store(node.relation)
        for rowid, row in store.scan():
            enriched = {"_rowid": rowid, **row}
            if node.predicate:
                if not _eval_expr(node.predicate, enriched):
                    continue
            yield enriched

    def _index_scan(self, node: IndexScanNode) -> RowIter:
        obj = self._catalog.get_table(node.relation)
        if obj is None:
            return
        idx = obj.get_component(node.index_name)
        if not isinstance(idx, IndexComponent):
            # Fall back to seq scan with filter
            fake_seq = SeqScanNode(
                relation=node.relation,
                predicate=node.predicate,
            )
            yield from self._seq_scan(fake_seq)
            return
        row_ids = idx.lookup(node.lookup_value)
        store = self._get_store(node.relation)
        for rid in row_ids:
            row = store.get(int(rid))
            if row is not None:
                yield {"_rowid": rid, **row}

    # ------------------------------------------------------------------
    # Relational operators
    # ------------------------------------------------------------------

    def _filter(self, node: FilterNode) -> RowIter:
        child_iter, _ = self._execute_node(node.child)
        for row in child_iter:
            if _eval_expr(node.predicate, row):
                yield row

    def _project(self, node: ProjectionNode) -> RowIter:
        cols = node.columns

        # Fast path: SELECT *
        if cols == ["*"]:
            child_iter, _ = self._execute_node(node.child)
            for row in child_iter:
                yield {k: v for k, v in row.items() if not k.startswith("_")}
            return

        # Detect whether any column expression is an aggregate function
        _AGG_RE = re.compile(r"(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(\*|\w+)\s*\)", re.IGNORECASE)
        has_aggregate = any(_AGG_RE.search(c) for c in cols)

        if has_aggregate:
            # Materialise all child rows once, then compute aggregates and
            # yield a single result row (full-table aggregation).
            child_iter, _ = self._execute_node(node.child)
            all_rows = list(child_iter)
            result: Row = {}
            for col_expr in cols:
                if " AS " in col_expr.upper():
                    parts = re.split(r"\s+AS\s+", col_expr, flags=re.IGNORECASE, maxsplit=1)
                    expr_text, alias = parts[0].strip(), parts[1].strip()
                else:
                    expr_text, alias = col_expr.strip(), col_expr.strip()
                agg_m = _AGG_RE.match(expr_text)
                if agg_m:
                    result[alias] = self._aggregate_rows(
                        agg_m.group(1).upper(), agg_m.group(2), all_rows
                    )
                elif all_rows:
                    row0 = all_rows[0]
                    result[alias] = row0.get(expr_text, _eval_expr(expr_text, row0))
                else:
                    result[alias] = None
            yield result
        else:
            # Row-by-row projection (no aggregates)
            child_iter, _ = self._execute_node(node.child)
            for row in child_iter:
                result = {}
                for col_expr in cols:
                    if " AS " in col_expr.upper():
                        parts = re.split(r"\s+AS\s+", col_expr, flags=re.IGNORECASE, maxsplit=1)
                        expr_text, alias = parts[0].strip(), parts[1].strip()
                    else:
                        expr_text, alias = col_expr.strip(), col_expr.strip()
                    if expr_text in row:
                        result[alias] = row[expr_text]
                    else:
                        try:
                            result[alias] = _eval_expr(expr_text, row)
                        except Exception:
                            result[alias] = None
                yield result

    def _aggregate_rows(self, func: str, col: str, rows: List[Row]) -> Any:
        """Compute an aggregate over an already-materialised list of rows."""
        if func == "COUNT":
            return len(rows) if col == "*" else sum(1 for r in rows if r.get(col) is not None)
        vals = [r.get(col) for r in rows if r.get(col) is not None]
        if not vals:
            return None
        if func == "SUM":
            return sum(vals)
        if func == "AVG":
            return sum(vals) / len(vals)
        if func == "MIN":
            return min(vals)
        if func == "MAX":
            return max(vals)
        return None

    def _aggregate(self, func: str, col: str, child_node: PlanNode) -> Any:
        """Compute an aggregate by re-executing the child plan (legacy helper)."""
        child_iter, _ = self._execute_node(child_node)
        return self._aggregate_rows(func, col, list(child_iter))

    def _sort(self, node: SortNode) -> RowIter:
        child_iter, _ = self._execute_node(node.child)
        rows = list(child_iter)
        for col, direction in reversed(node.order_by):
            reverse = direction.upper() == "DESC"
            rows.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=reverse)
        yield from rows

    def _limit(self, node: LimitNode) -> RowIter:
        child_iter, _ = self._execute_node(node.child)
        offset = node.offset or 0
        limit = node.limit
        count = 0
        for row in child_iter:
            if offset > 0:
                offset -= 1
                continue
            if limit is not None and count >= limit:
                break
            yield row
            count += 1

    def _nested_loop_join(self, node: NestedLoopJoinNode) -> RowIter:
        outer_iter, _ = self._execute_node(node.outer)
        outer_rows = list(outer_iter)
        for outer_row in outer_rows:
            inner_iter, _ = self._execute_node(node.inner)
            for inner_row in inner_iter:
                combined = {**outer_row, **inner_row}
                if node.condition:
                    if not _eval_expr(node.condition, combined):
                        if node.join_type == "INNER":
                            continue
                        # LEFT join: include outer row with NULLs for inner
                        if node.join_type == "LEFT":
                            yield {**outer_row, **{k: None for k in inner_row}}
                            continue
                yield combined

    # ------------------------------------------------------------------
    # DML
    # ------------------------------------------------------------------

    def _insert(self, node: InsertNode) -> int:
        obj = self._catalog.get_table(node.relation)
        if obj is None:
            raise ValueError(f"Table '{node.relation}' does not exist")
        col_comp = obj.get_component_typed(ColumnSetComponent)
        store = self._get_store(node.relation)
        count = 0
        for value_list in node.values:
            row: Row = {}
            if col_comp and node.columns:
                for col_name, val in zip(node.columns, value_list):
                    col_def = col_comp.get(col_name)
                    row[col_name] = self._coerce(val, col_def)
            else:
                row = {c: v for c, v in zip(node.columns, value_list)}

            # CHECK constraints
            for comp in obj.all_components():
                if isinstance(comp, CheckComponent):
                    if not comp.evaluate(row):
                        raise ValueError(f"CHECK constraint '{comp.name}' violated")

            # BEFORE INSERT triggers
            for comp in obj.all_components():
                if isinstance(comp, TriggerComponent) and comp.timing == "BEFORE":
                    comp.fire("INSERT", None, row)

            rowid = store.insert(row)

            # Update indexes
            for comp in obj.all_components():
                if isinstance(comp, IndexComponent):
                    key_val = tuple(row.get(c) for c in comp.columns)
                    comp.insert(key_val if len(comp.columns) > 1 else key_val[0], rowid)

            # AFTER INSERT triggers
            for comp in obj.all_components():
                if isinstance(comp, TriggerComponent) and comp.timing == "AFTER":
                    comp.fire("INSERT", None, row)

            count += 1
        self._catalog.update_stats(node.relation, row_delta=count)
        return count

    def _update(self, node: UpdateNode) -> int:
        obj = self._catalog.get_table(node.relation)
        if obj is None:
            raise ValueError(f"Table '{node.relation}' does not exist")
        store = self._get_store(node.relation)
        count = 0
        for rowid, row in list(store.scan()):
            enriched = {"_rowid": rowid, **row}
            if node.predicate and not _eval_expr(node.predicate, enriched):
                continue
            new_row = {**row, **node.assignments}
            store.update(rowid, node.assignments)

            # Update indexes
            for comp in obj.all_components():
                if isinstance(comp, IndexComponent):
                    old_key = tuple(row.get(c) for c in comp.columns)
                    new_key = tuple(new_row.get(c) for c in comp.columns)
                    if old_key != new_key:
                        comp.delete(
                            old_key if len(comp.columns) > 1 else old_key[0], rowid
                        )
                        comp.insert(
                            new_key if len(comp.columns) > 1 else new_key[0], rowid
                        )
            count += 1
        return count

    def _delete(self, node: DeleteNode) -> int:
        obj = self._catalog.get_table(node.relation)
        if obj is None:
            raise ValueError(f"Table '{node.relation}' does not exist")
        store = self._get_store(node.relation)
        to_delete = []
        for rowid, row in store.scan():
            enriched = {"_rowid": rowid, **row}
            if node.predicate and not _eval_expr(node.predicate, enriched):
                continue
            to_delete.append((rowid, row))

        for rowid, row in to_delete:
            for comp in obj.all_components():
                if isinstance(comp, IndexComponent):
                    key = tuple(row.get(c) for c in comp.columns)
                    comp.delete(key if len(comp.columns) > 1 else key[0], rowid)
            store.delete(rowid)

        self._catalog.update_stats(node.relation, row_delta=-len(to_delete))
        return len(to_delete)

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------

    def _create_table(self, node: CreateTableNode) -> int:
        stmt = node.stmt
        schema = {
            "name": stmt.table_name,
            "type": "table",
            "columns": [
                {
                    "name": col.name,
                    "type": col.col_type,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "default": col.default,
                }
                for col in stmt.columns
            ],
        }
        try:
            self._catalog.create_table(schema)
        except ValueError:
            if not stmt.if_not_exists:
                raise
        return 0

    def _drop_table(self, node: DropTableNode) -> int:
        stmt = node.stmt
        try:
            self._catalog.drop_table(stmt.table_name)
            self._stores.pop(stmt.table_name, None)
        except KeyError:
            if not stmt.if_exists:
                raise
        return 0

    def _create_index(self, node: CreateIndexNode) -> int:
        stmt = node.stmt
        obj = self._catalog.get_table(stmt.table_name)
        if obj is None:
            raise ValueError(f"Table '{stmt.table_name}' does not exist")
        from llmdb.core.actors import IndexComponent
        idx = IndexComponent(
            index_name=stmt.index_name,
            columns=stmt.columns,
            unique=stmt.unique,
        )
        try:
            obj.add_component(idx)
        except ValueError:
            pass  # already exists
        return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_store(self, relation: str) -> _RowStore:
        if relation not in self._stores:
            self._stores[relation] = _RowStore()
        return self._stores[relation]

    def _coerce(self, value: Any, col_def: Any) -> Any:
        if col_def is None or value is None:
            return value
        from llmdb.core.actors import ColType
        try:
            if col_def.col_type == ColType.INTEGER:
                return int(value)
            elif col_def.col_type == ColType.REAL:
                return float(value)
            elif col_def.col_type == ColType.TEXT:
                return str(value)
            elif col_def.col_type == ColType.BOOLEAN:
                return bool(value)
        except (ValueError, TypeError):
            pass
        return value
