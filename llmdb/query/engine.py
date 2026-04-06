"""Query engine — structured filter-based queries over a Collection.

The query language is intentionally simple and JSON-serialisable so that
an LLM can construct queries programmatically without learning SQL.

Supported filter operators
--------------------------
``$eq``  — field equals value
``$ne``  — field not equal to value
``$gt``  — field greater than value
``$gte`` — field greater than or equal to value
``$lt``  — field less than value
``$lte`` — field less than or equal to value
``$in``  — field value is in a list
``$contains`` — string field contains substring (case-insensitive)
``$exists``   — field is present (bool)

Logical combinators
-------------------
``$and`` — all sub-filters must match
``$or``  — at least one sub-filter must match
``$not`` — sub-filter must NOT match

Example::

    {
        "$and": [
            {"role": {"$eq": "assistant"}},
            {"token_count": {"$lt": 500}},
            {"$or": [
                {"model": {"$eq": "gpt-4"}},
                {"model": {"$eq": "claude-3"}},
            ]}
        ]
    }
"""

from __future__ import annotations

from typing import Any

from llmdb.core.document import Document
from llmdb.exceptions import QueryError


# ---------------------------------------------------------------------------
# Filter evaluation
# ---------------------------------------------------------------------------


def _get_nested(data: dict[str, Any], field: str) -> Any:
    """Support dot-separated nested field access (e.g. ``"meta.author"``)."""
    parts = field.split(".")
    cur: Any = data
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _eval_field_op(value: Any, op: str, operand: Any) -> bool:
    """Evaluate a single field-level operator."""
    if op == "$eq":
        return value == operand
    if op == "$ne":
        return value != operand
    if op == "$gt":
        return value is not None and value > operand
    if op == "$gte":
        return value is not None and value >= operand
    if op == "$lt":
        return value is not None and value < operand
    if op == "$lte":
        return value is not None and value <= operand
    if op == "$in":
        if not isinstance(operand, list):
            raise QueryError(f"$in requires a list operand, got {type(operand)}")
        return value in operand
    if op == "$contains":
        return (
            isinstance(value, str)
            and isinstance(operand, str)
            and operand.lower() in value.lower()
        )
    if op == "$exists":
        return (value is not None) == bool(operand)
    raise QueryError(f"Unknown operator: {op!r}")


def _eval_filter(doc_data: dict[str, Any], filt: dict[str, Any]) -> bool:
    """Recursively evaluate a filter dict against document data."""
    for key, value in filt.items():
        if key == "$and":
            if not isinstance(value, list):
                raise QueryError("$and requires a list of sub-filters.")
            if not all(_eval_filter(doc_data, sub) for sub in value):
                return False
        elif key == "$or":
            if not isinstance(value, list):
                raise QueryError("$or requires a list of sub-filters.")
            if not any(_eval_filter(doc_data, sub) for sub in value):
                return False
        elif key == "$not":
            if not isinstance(value, dict):
                raise QueryError("$not requires a dict sub-filter.")
            if _eval_filter(doc_data, value):
                return False
        else:
            # key is a field name
            field_val = _get_nested(doc_data, key)
            if isinstance(value, dict) and any(
                k.startswith("$") for k in value
            ):
                # operator dict, e.g. {"$gt": 5}
                for op, operand in value.items():
                    if not _eval_field_op(field_val, op, operand):
                        return False
            else:
                # shorthand equality: {"field": value}
                if field_val != value:
                    return False
    return True


# ---------------------------------------------------------------------------
# QueryEngine
# ---------------------------------------------------------------------------


class QueryEngine:
    """Execute structured queries against a collection's documents."""

    def __init__(self, storage: Any, collection_name: str) -> None:
        self._storage = storage
        self._collection = collection_name

    def execute(
        self,
        filter: dict[str, Any] | None = None,
        *,
        sort_by: str | None = None,
        descending: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """Run a query and return matching documents.

        Args:
            filter: A filter dict (see module docstring).  ``None`` means
                    "return all documents".
            sort_by: A dot-separated field path to sort results by.  Sorting
                     operates on the final Python values (in-process).
            descending: Sort direction (default: descending).
            limit: Maximum number of results to return.
            offset: Number of results to skip before returning.

        Returns:
            List of :class:`~llmdb.core.document.Document` objects.
        """
        import json

        rows = self._storage.execute(
            "SELECT id, collection, data, tags, score, created_at, updated_at "
            "FROM documents WHERE collection = ?",
            (self._collection,),
        ).fetchall()

        results: list[Document] = []
        for row in rows:
            data = json.loads(row["data"])
            if filter is None or _eval_filter(data, filter):
                results.append(
                    Document(
                        data=data,
                        id=row["id"],
                        collection=row["collection"],
                        tags=json.loads(row["tags"]),
                        score=row["score"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )

        if sort_by is not None:
            results.sort(
                key=lambda d: (_get_nested(d.data, sort_by) is None,
                               _get_nested(d.data, sort_by)),
                reverse=descending,
            )

        return results[offset: offset + limit]

    def count(self, filter: dict[str, Any] | None = None) -> int:
        """Return the number of documents matching a filter."""
        return len(self.execute(filter, limit=10_000_000))
