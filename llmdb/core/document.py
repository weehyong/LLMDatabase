"""Document model — the fundamental unit of data in LLMDatabase."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class Document:
    """A flexible, schema-less record stored in a Collection.

    Agents work with arbitrary JSON payloads; no upfront schema is required.
    Every document carries a unique ID, tags for quick retrieval, an optional
    importance *score*, and creation/update timestamps.
    """

    def __init__(
        self,
        data: dict[str, Any],
        *,
        id: str | None = None,
        collection: str = "",
        tags: list[str] | None = None,
        score: float = 0.0,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.collection = collection
        self.data: dict[str, Any] = data
        self.tags: list[str] = tags or []
        self.score = score
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "collection": self.collection,
            "data": self.data,
            "tags": self.tags,
            "score": self.score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        return cls(
            data=d["data"],
            id=d.get("id"),
            collection=d.get("collection", ""),
            tags=d.get("tags", []),
            score=d.get("score", 0.0),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
        )

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, collection={self.collection!r})"
