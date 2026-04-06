"""Collection — a named bucket of Documents with optional vector support."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llmdb.core.document import Document
from llmdb.core.storage import StorageEngine
from llmdb.exceptions import (
    CollectionExistsError,
    CollectionNotFoundError,
    DocumentNotFoundError,
    VectorDimensionError,
)


class Collection:
    """Manages all documents (and their embeddings) inside one collection.

    A collection is analogous to a table in a relational database, but it
    stores JSON documents and—optionally—fixed-dimension embedding vectors
    alongside each document for semantic search.
    """

    def __init__(
        self,
        name: str,
        storage: StorageEngine,
        *,
        vector_dim: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self._storage = storage
        self.vector_dim = vector_dim
        self.metadata: dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Factory methods (used by the Engine)
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        storage: StorageEngine,
        *,
        vector_dim: int | None = None,
        metadata: dict[str, Any] | None = None,
        exist_ok: bool = False,
    ) -> "Collection":
        """Persist a new collection and return the Collection object."""
        row = storage.execute(
            "SELECT name FROM collections WHERE name = ?", (name,)
        ).fetchone()
        if row:
            if exist_ok:
                return cls.load(name, storage)
            raise CollectionExistsError(f"Collection '{name}' already exists.")

        meta_str = storage.encode(metadata or {})
        storage.execute(
            "INSERT INTO collections (name, metadata, vector_dim) VALUES (?, ?, ?)",
            (name, meta_str, vector_dim),
            commit=True,
        )
        return cls(name, storage, vector_dim=vector_dim, metadata=metadata or {})

    @classmethod
    def load(cls, name: str, storage: StorageEngine) -> "Collection":
        """Load an existing collection from storage."""
        row = storage.execute(
            "SELECT metadata, vector_dim FROM collections WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            raise CollectionNotFoundError(f"Collection '{name}' not found.")
        return cls(
            name,
            storage,
            vector_dim=row["vector_dim"],
            metadata=storage.decode(row["metadata"]),
        )

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def insert(
        self,
        data: dict[str, Any],
        *,
        id: str | None = None,
        tags: list[str] | None = None,
        score: float = 0.0,
        embedding: list[float] | None = None,
    ) -> Document:
        """Insert a new document and optionally store its embedding."""
        doc = Document(data, id=id, collection=self.name, tags=tags, score=score)
        self._store_document(doc)
        if embedding is not None:
            self._store_vector(doc.id, embedding)
        return doc

    def upsert(
        self,
        data: dict[str, Any],
        *,
        id: str | None = None,
        tags: list[str] | None = None,
        score: float = 0.0,
        embedding: list[float] | None = None,
    ) -> Document:
        """Insert or replace a document by ID."""
        if id is None:
            return self.insert(data, tags=tags, score=score, embedding=embedding)
        try:
            self.delete(id)
        except DocumentNotFoundError:
            pass
        return self.insert(data, id=id, tags=tags, score=score, embedding=embedding)

    def get(self, doc_id: str) -> Document:
        """Retrieve a single document by ID."""
        row = self._storage.execute(
            "SELECT id, collection, data, tags, score, created_at, updated_at "
            "FROM documents WHERE id = ? AND collection = ?",
            (doc_id, self.name),
        ).fetchone()
        if not row:
            raise DocumentNotFoundError(
                f"Document '{doc_id}' not found in '{self.name}'."
            )
        return self._row_to_doc(row)

    def update(self, doc_id: str, data: dict[str, Any]) -> Document:
        """Merge *data* into an existing document."""
        doc = self.get(doc_id)
        doc.data.update(data)
        doc.updated_at = datetime.now(timezone.utc).isoformat()
        self._storage.execute(
            "UPDATE documents SET data = ?, updated_at = ? "
            "WHERE id = ? AND collection = ?",
            (
                self._storage.encode(doc.data),
                doc.updated_at,
                doc_id,
                self.name,
            ),
            commit=True,
        )
        return doc

    def delete(self, doc_id: str) -> None:
        """Delete a document (and its vector if present)."""
        row = self._storage.execute(
            "SELECT id FROM documents WHERE id = ? AND collection = ?",
            (doc_id, self.name),
        ).fetchone()
        if not row:
            raise DocumentNotFoundError(
                f"Document '{doc_id}' not found in '{self.name}'."
            )
        self._storage.execute(
            "DELETE FROM documents WHERE id = ? AND collection = ?",
            (doc_id, self.name),
            commit=True,
        )

    def list(
        self,
        *,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents, optionally filtered by tags.

        When *tags* are provided, filtering is performed in Python because tags
        are stored as a JSON array (not individual indexed rows).  All rows are
        fetched up-front so that *offset* and *limit* are applied over the
        complete filtered result set, ensuring correct pagination.
        """
        if tags:
            rows = self._storage.execute(
                "SELECT id, collection, data, tags, score, created_at, updated_at "
                "FROM documents WHERE collection = ? "
                "ORDER BY created_at DESC",
                (self.name,),
            ).fetchall()
            docs = [self._row_to_doc(r) for r in rows]
            filtered = [d for d in docs if any(t in d.tags for t in tags)]
            return filtered[offset: offset + limit]
        rows = self._storage.execute(
            "SELECT id, collection, data, tags, score, created_at, updated_at "
            "FROM documents WHERE collection = ? "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (self.name, limit, offset),
        ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def count(self) -> int:
        """Return the total number of documents in this collection."""
        row = self._storage.execute(
            "SELECT COUNT(*) AS c FROM documents WHERE collection = ?",
            (self.name,),
        ).fetchone()
        return row["c"] if row else 0

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def set_embedding(self, doc_id: str, embedding: list[float]) -> None:
        """Store or replace the embedding for an existing document."""
        self.get(doc_id)  # raises if not found
        self._store_vector(doc_id, embedding)

    def get_embedding(self, doc_id: str) -> list[float] | None:
        """Return the raw embedding for a document, or None."""
        row = self._storage.execute(
            "SELECT embedding FROM vectors WHERE doc_id = ? AND collection = ?",
            (doc_id, self.name),
        ).fetchone()
        return self._storage.decode(row["embedding"]) if row else None

    def _store_vector(self, doc_id: str, embedding: list[float]) -> None:
        if self.vector_dim is not None and len(embedding) != self.vector_dim:
            raise VectorDimensionError(
                f"Expected {self.vector_dim}-dim vector, got {len(embedding)}."
            )
        self._storage.execute(
            "INSERT OR REPLACE INTO vectors (doc_id, collection, embedding) "
            "VALUES (?, ?, ?)",
            (doc_id, self.name, self._storage.encode(embedding)),
            commit=True,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_document(self, doc: Document) -> None:
        self._storage.execute(
            "INSERT INTO documents "
            "(id, collection, data, tags, score, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                doc.id,
                doc.collection,
                self._storage.encode(doc.data),
                self._storage.encode(doc.tags),
                doc.score,
                doc.created_at,
                doc.updated_at,
            ),
            commit=True,
        )

    @staticmethod
    def _row_to_doc(row: Any) -> Document:
        import json

        return Document(
            data=json.loads(row["data"]),
            id=row["id"],
            collection=row["collection"],
            tags=json.loads(row["tags"]),
            score=row["score"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def __repr__(self) -> str:
        return f"Collection(name={self.name!r}, vector_dim={self.vector_dim!r})"
