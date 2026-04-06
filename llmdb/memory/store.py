"""Memory store — persistent, typed memories for AI agents.

Agents operate with three kinds of memory:

* **Episodic** — specific events and experiences (e.g. "The user asked me
  to book a flight to Paris on 2025-03-12").
* **Semantic** — general facts and world knowledge (e.g. "Paris is the
  capital of France").
* **Procedural** — how-to knowledge and skills (e.g. "To book a flight,
  call the /book-flight API with origin, destination, and date").

All memories are persisted in SQLite and can optionally carry an embedding
for semantic retrieval.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from llmdb.core.storage import StorageEngine
from llmdb.exceptions import DocumentNotFoundError
from llmdb.vector.index import VectorIndex, _cosine_similarity


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Memory:
    """A single memory entry."""

    def __init__(
        self,
        content: str,
        memory_type: MemoryType | str = MemoryType.EPISODIC,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        created_at: str | None = None,
        accessed_at: str | None = None,
        access_count: int = 0,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.memory_type = MemoryType(memory_type)
        self.metadata: dict[str, Any] = metadata or {}
        self.importance = max(0.0, min(1.0, importance))
        self.tags: list[str] = tags or []
        self.embedding: list[float] | None = embedding
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = created_at or now
        self.accessed_at = accessed_at or now
        self.access_count = access_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "importance": self.importance,
            "tags": self.tags,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Memory":
        return cls(
            content=d["content"],
            memory_type=d.get("memory_type", MemoryType.EPISODIC),
            id=d.get("id"),
            metadata=d.get("metadata", {}),
            importance=d.get("importance", 0.5),
            tags=d.get("tags", []),
            embedding=d.get("embedding"),
            created_at=d.get("created_at"),
            accessed_at=d.get("accessed_at"),
            access_count=d.get("access_count", 0),
        )

    def __repr__(self) -> str:
        return (
            f"Memory(id={self.id!r}, type={self.memory_type.value!r}, "
            f"importance={self.importance:.2f})"
        )


class MemoryStore:
    """Persistent store for all three memory types.

    Provides semantic retrieval (when embeddings are present), importance-
    based retrieval, and tag-based filtering — the access patterns agents
    need most.
    """

    def __init__(self, storage: StorageEngine) -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        memory_type: MemoryType | str = MemoryType.EPISODIC,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> Memory:
        """Persist a new memory and return it."""
        mem = Memory(
            content=content,
            memory_type=memory_type,
            id=id,
            metadata=metadata,
            importance=importance,
            tags=tags,
            embedding=embedding,
        )
        self._storage.execute(
            "INSERT INTO memories "
            "(id, memory_type, content, metadata, importance, tags, embedding, "
            " created_at, accessed_at, access_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mem.id,
                mem.memory_type.value,
                mem.content,
                self._storage.encode(mem.metadata),
                mem.importance,
                self._storage.encode(mem.tags),
                self._storage.encode(mem.embedding) if mem.embedding else None,
                mem.created_at,
                mem.accessed_at,
                mem.access_count,
            ),
            commit=True,
        )
        return mem

    def update_importance(self, memory_id: str, importance: float) -> Memory:
        """Update the importance score of an existing memory."""
        importance = max(0.0, min(1.0, importance))
        self.get(memory_id)  # raises if not found
        self._storage.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (importance, memory_id),
            commit=True,
        )
        return self.get(memory_id)

    def delete(self, memory_id: str) -> None:
        """Remove a memory by ID."""
        row = self._storage.execute(
            "SELECT id FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            raise DocumentNotFoundError(f"Memory '{memory_id}' not found.")
        self._storage.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,), commit=True
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Memory:
        """Retrieve a memory by ID and increment its access counter."""
        row = self._storage.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            raise DocumentNotFoundError(f"Memory '{memory_id}' not found.")
        mem = self._row_to_memory(row)
        # Update access tracking
        now = datetime.now(timezone.utc).isoformat()
        self._storage.execute(
            "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 "
            "WHERE id = ?",
            (now, memory_id),
            commit=True,
        )
        return mem

    def list(
        self,
        *,
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories with optional filters, ordered by importance desc."""
        sql = (
            "SELECT * FROM memories WHERE importance >= ? "
        )
        params: list[Any] = [min_importance]

        if memory_type is not None:
            sql += "AND memory_type = ? "
            params.append(MemoryType(memory_type).value)

        sql += "ORDER BY importance DESC, created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit + (len(tags or [])) * 10, offset])

        rows = self._storage.execute(sql, tuple(params)).fetchall()
        mems = [self._row_to_memory(r) for r in rows]

        if tags:
            mems = [m for m in mems if any(t in m.tags for t in tags)]

        return mems[: limit]

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        threshold: float = 0.0,
        memory_type: MemoryType | str | None = None,
    ) -> list[tuple[Memory, float]]:
        """Semantic search: return (Memory, similarity) pairs ordered by score.

        Only memories that have stored embeddings participate.
        """
        sql = "SELECT * FROM memories WHERE embedding IS NOT NULL"
        params: list[Any] = []
        if memory_type is not None:
            sql += " AND memory_type = ?"
            params.append(MemoryType(memory_type).value)

        rows = self._storage.execute(sql, tuple(params)).fetchall()
        scored: list[tuple[Memory, float]] = []
        for row in rows:
            mem = self._row_to_memory(row)
            if mem.embedding is None:
                continue
            sim = _cosine_similarity(query_embedding, mem.embedding)
            if sim >= threshold:
                scored.append((mem, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def count(self, memory_type: MemoryType | str | None = None) -> int:
        """Return the total number of stored memories, optionally by type."""
        if memory_type is not None:
            row = self._storage.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE memory_type = ?",
                (MemoryType(memory_type).value,),
            ).fetchone()
        else:
            row = self._storage.execute(
                "SELECT COUNT(*) AS c FROM memories"
            ).fetchone()
        return row["c"] if row else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _row_to_memory(self, row: Any) -> Memory:
        import json

        return Memory(
            content=row["content"],
            memory_type=row["memory_type"],
            id=row["id"],
            metadata=json.loads(row["metadata"]),
            importance=row["importance"],
            tags=json.loads(row["tags"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            created_at=row["created_at"],
            accessed_at=row["accessed_at"],
            access_count=row["access_count"],
        )
