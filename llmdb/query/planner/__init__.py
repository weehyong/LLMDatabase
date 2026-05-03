"""
Query Planner — cost-based query planning.

Inspired by OpenClaw's 'PathfinderComponent' which computed the cheapest
path through the level for an enemy AI.  The query planner does the same
for data retrieval: it examines available indexes and statistics, then
chooses the cheapest access path.

Plan nodes form a tree; the executor walks the tree iterating tuples upward
(the Volcano model).  Each node is a lazy iterator that pulls from its child.

Supported plan nodes
---------------------
  SeqScan          — full heap scan
  IndexScan        — B-tree index point/range lookup
  Filter           — predicate evaluation on top of any node
  Projection       — column selection / expression evaluation
  NestedLoopJoin   — simple nested-loop join
  Sort             — in-memory sort
  Limit            — top-N truncation
  InsertNode       — DML insert
  UpdateNode       — DML update
  DeleteNode       — DML delete
  CreateTableNode  — DDL
  DropTableNode    — DDL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llmdb.query.sql import (
    CreateTableStmt, DropTableStmt, CreateIndexStmt, DropIndexStmt,
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    BeginStmt, CommitStmt, RollbackStmt,
    Stmt,
)


# ---------------------------------------------------------------------------
# Plan Nodes
# ---------------------------------------------------------------------------

@dataclass
class PlanNode:
    """Base plan node."""
    estimated_rows: int = 0
    estimated_cost: float = 0.0


@dataclass
class SeqScanNode(PlanNode):
    relation: str = ""
    alias: Optional[str] = None
    predicate: Optional[str] = None   # WHERE expression text


@dataclass
class IndexScanNode(PlanNode):
    relation: str = ""
    index_name: str = ""
    key_column: str = ""
    lookup_value: Any = None
    predicate: Optional[str] = None


@dataclass
class FilterNode(PlanNode):
    child: Optional[PlanNode] = None
    predicate: str = ""


@dataclass
class ProjectionNode(PlanNode):
    child: Optional[PlanNode] = None
    columns: List[str] = field(default_factory=list)   # ["*"] or explicit names


@dataclass
class SortNode(PlanNode):
    child: Optional[PlanNode] = None
    order_by: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class LimitNode(PlanNode):
    child: Optional[PlanNode] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class NestedLoopJoinNode(PlanNode):
    outer: Optional[PlanNode] = None
    inner: Optional[PlanNode] = None
    join_type: str = "INNER"
    condition: Optional[str] = None


@dataclass
class InsertNode(PlanNode):
    relation: str = ""
    columns: List[str] = field(default_factory=list)
    values: List[List[Any]] = field(default_factory=list)


@dataclass
class UpdateNode(PlanNode):
    relation: str = ""
    assignments: Dict[str, Any] = field(default_factory=dict)
    predicate: Optional[str] = None


@dataclass
class DeleteNode(PlanNode):
    relation: str = ""
    predicate: Optional[str] = None


@dataclass
class CreateTableNode(PlanNode):
    stmt: Optional[CreateTableStmt] = None


@dataclass
class DropTableNode(PlanNode):
    stmt: Optional[DropTableStmt] = None


@dataclass
class CreateIndexNode(PlanNode):
    stmt: Optional[CreateIndexStmt] = None


@dataclass
class DropIndexNode(PlanNode):
    stmt: Optional[DropIndexStmt] = None


@dataclass
class TransactionNode(PlanNode):
    operation: str = "BEGIN"  # BEGIN | COMMIT | ROLLBACK


@dataclass
class NoOpNode(PlanNode):
    pass


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """Translates an AST into an executable plan tree.

    The planner consults a lightweight Catalog (injected) to check which
    indexes exist and to estimate row counts for cost calculations.

    Cost model (simplified)
    -----------------------
      SeqScan cost  = page_count * 1.0
      IndexScan cost = log2(row_count) * 0.1   (O(log n) B-tree lookup)
      Sort cost     = n * log(n) * 0.01
      NestedLoop    = outer_cost + outer_rows * inner_cost
    """

    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog

    # ------------------------------------------------------------------

    def plan(self, ast: Stmt) -> PlanNode:
        if isinstance(ast, SelectStmt):
            return self._plan_select(ast)
        elif isinstance(ast, InsertStmt):
            return InsertNode(relation=ast.table_name, columns=ast.columns, values=ast.values)
        elif isinstance(ast, UpdateStmt):
            return UpdateNode(
                relation=ast.table_name,
                assignments=ast.assignments,
                predicate=ast.where.text if ast.where else None,
            )
        elif isinstance(ast, DeleteStmt):
            return DeleteNode(
                relation=ast.table_name,
                predicate=ast.where.text if ast.where else None,
            )
        elif isinstance(ast, CreateTableStmt):
            return CreateTableNode(stmt=ast)
        elif isinstance(ast, DropTableStmt):
            return DropTableNode(stmt=ast)
        elif isinstance(ast, CreateIndexStmt):
            return CreateIndexNode(stmt=ast)
        elif isinstance(ast, DropIndexStmt):
            return DropIndexNode(stmt=ast)
        elif isinstance(ast, BeginStmt):
            return TransactionNode(operation="BEGIN")
        elif isinstance(ast, CommitStmt):
            return TransactionNode(operation="COMMIT")
        elif isinstance(ast, RollbackStmt):
            return TransactionNode(operation="ROLLBACK")
        return NoOpNode()

    # ------------------------------------------------------------------
    # SELECT planning
    # ------------------------------------------------------------------

    def _plan_select(self, stmt: SelectStmt) -> PlanNode:
        # 1. Base access node (FROM clause)
        if stmt.from_table is None:
            node: PlanNode = NoOpNode()
        else:
            node = self._best_access_path(
                stmt.from_table, stmt.table_alias, stmt.where
            )

        # 2. JOINs
        for join in stmt.joins:
            inner = self._best_access_path(join.table_name, join.alias, join.condition)
            outer_rows = getattr(node, "estimated_rows", 1000)
            inner_rows = getattr(inner, "estimated_rows", 1000)
            node = NestedLoopJoinNode(
                outer=node,
                inner=inner,
                join_type=join.join_type,
                condition=join.condition.text if join.condition else None,
                estimated_rows=outer_rows * inner_rows // 10,
                estimated_cost=(
                    getattr(node, "estimated_cost", 1.0)
                    + outer_rows * getattr(inner, "estimated_cost", 1.0)
                ),
            )

        # 3. Filter (if WHERE was not pushed into the scan)
        if stmt.where and not isinstance(node, (SeqScanNode, IndexScanNode)):
            node = FilterNode(
                child=node,
                predicate=stmt.where.text,
                estimated_rows=max(1, getattr(node, "estimated_rows", 100) // 3),
            )

        # 4. GROUP BY (placeholder — not fully implemented)

        # 5. HAVING
        if stmt.having:
            node = FilterNode(
                child=node,
                predicate=stmt.having.text,
                estimated_rows=max(1, getattr(node, "estimated_rows", 100) // 2),
            )

        # 6. Projection
        node = ProjectionNode(
            child=node,
            columns=stmt.columns,
            estimated_rows=getattr(node, "estimated_rows", 0),
        )

        # 7. ORDER BY
        if stmt.order_by:
            n = getattr(node, "estimated_rows", 100)
            import math
            sort_cost = n * math.log2(max(n, 1)) * 0.01
            node = SortNode(
                child=node,
                order_by=stmt.order_by,
                estimated_rows=n,
                estimated_cost=sort_cost,
            )

        # 8. LIMIT / OFFSET
        if stmt.limit is not None or stmt.offset is not None:
            node = LimitNode(
                child=node,
                limit=stmt.limit,
                offset=stmt.offset,
                estimated_rows=min(
                    stmt.limit or 2**31,
                    getattr(node, "estimated_rows", 0),
                ),
            )

        return node

    # ------------------------------------------------------------------

    def _best_access_path(
        self, table: str, alias: Optional[str], where_expr: Any
    ) -> PlanNode:
        """Choose between SeqScan and IndexScan."""
        row_count = self._catalog.estimated_row_count(table)
        page_count = max(1, row_count // 100)

        seq_cost = float(page_count)

        # Try to find an index that can satisfy the WHERE predicate
        if where_expr is not None:
            index = self._catalog.find_index_for_predicate(table, where_expr.text)
            if index is not None:
                import math
                idx_cost = math.log2(max(row_count, 2)) * 0.1
                if idx_cost < seq_cost:
                    return IndexScanNode(
                        relation=table,
                        index_name=index["name"],
                        key_column=index["column"],
                        lookup_value=index.get("value"),
                        predicate=where_expr.text,
                        estimated_rows=max(1, row_count // 10),
                        estimated_cost=idx_cost,
                    )

        return SeqScanNode(
            relation=table,
            alias=alias,
            predicate=where_expr.text if where_expr else None,
            estimated_rows=row_count,
            estimated_cost=seq_cost,
        )
