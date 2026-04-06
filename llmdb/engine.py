"""LLMDatabase Engine — the top-level façade for all operations.

Usage::

    from llmdb import Database

    db = Database("mydb.llmdb")

    # Collections
    posts = db.create_collection("posts")
    doc   = posts.insert({"title": "Hello", "body": "World"}, tags=["blog"])

    # Semantic search
    posts.insert({"title": "…"}, embedding=[0.1, 0.9, …])
    results = db.vector_search("posts", query_embedding=[…], top_k=5)

    # Agent memory
    db.memory.add("User prefers dark mode", memory_type="semantic",
                  importance=0.8, tags=["preferences"])
    relevant = db.memory.search(query_embedding=[…], top_k=3)

    # Structured query
    docs = db.query("posts", {"author": {"$eq": "alice"}, "views": {"$gt": 100}})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llmdb.core.collection import Collection
from llmdb.core.document import Document
from llmdb.core.storage import StorageEngine
from llmdb.exceptions import CollectionNotFoundError
from llmdb.memory.store import Memory, MemoryStore, MemoryType
from llmdb.query.engine import QueryEngine
from llmdb.vector.index import SearchResult, VectorIndex


class Database:
    """The primary entry point for LLMDatabase.

    All operations — collections, documents, memory, and vector search —
    are accessible through a single ``Database`` instance.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        """Open (or create) a database at the given file path.

        Pass ``":memory:"`` for an in-memory database (testing / scratch work).
        """
        self._storage = StorageEngine(path)
        self.memory = MemoryStore(self._storage)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        name: str,
        *,
        vector_dim: int | None = None,
        metadata: dict[str, Any] | None = None,
        exist_ok: bool = False,
    ) -> Collection:
        """Create and return a new collection."""
        return Collection.create(
            name,
            self._storage,
            vector_dim=vector_dim,
            metadata=metadata,
            exist_ok=exist_ok,
        )

    def get_collection(self, name: str) -> Collection:
        """Return an existing collection by name."""
        return Collection.load(name, self._storage)

    def drop_collection(self, name: str) -> None:
        """Delete a collection and all its documents."""
        row = self._storage.execute(
            "SELECT name FROM collections WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            raise CollectionNotFoundError(f"Collection '{name}' not found.")
        self._storage.execute(
            "DELETE FROM collections WHERE name = ?", (name,), commit=True
        )

    def list_collections(self) -> list[dict[str, Any]]:
        """Return metadata for every collection in this database."""
        rows = self._storage.execute(
            "SELECT name, metadata, vector_dim, created_at FROM collections "
            "ORDER BY created_at"
        ).fetchall()
        return [
            {
                "name": r["name"],
                "metadata": self._storage.decode(r["metadata"]),
                "vector_dim": r["vector_dim"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def collection_exists(self, name: str) -> bool:
        """Return True if a collection with *name* exists."""
        row = self._storage.execute(
            "SELECT name FROM collections WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Convenience document operations
    # ------------------------------------------------------------------

    def insert(
        self,
        collection: str,
        data: dict[str, Any],
        *,
        id: str | None = None,
        tags: list[str] | None = None,
        score: float = 0.0,
        embedding: list[float] | None = None,
    ) -> Document:
        """Insert a document into *collection* (collection must exist)."""
        return self.get_collection(collection).insert(
            data, id=id, tags=tags, score=score, embedding=embedding
        )

    def get(self, collection: str, doc_id: str) -> Document:
        """Get a document from *collection*."""
        return self.get_collection(collection).get(doc_id)

    def update(self, collection: str, doc_id: str, data: dict[str, Any]) -> Document:
        """Merge *data* into a document in *collection*."""
        return self.get_collection(collection).update(doc_id, data)

    def delete(self, collection: str, doc_id: str) -> None:
        """Delete a document from *collection*."""
        self.get_collection(collection).delete(doc_id)

    # ------------------------------------------------------------------
    # Structured querying
    # ------------------------------------------------------------------

    def query(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
        *,
        sort_by: str | None = None,
        descending: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """Run a structured query against *collection*."""
        if not self.collection_exists(collection):
            raise CollectionNotFoundError(f"Collection '{collection}' not found.")
        engine = QueryEngine(self._storage, collection)
        return engine.execute(
            filter,
            sort_by=sort_by,
            descending=descending,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # Vector / semantic search
    # ------------------------------------------------------------------

    def vector_search(
        self,
        collection: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        threshold: float = -1.0,
        include_documents: bool = True,
    ) -> list[dict[str, Any]]:
        """Semantic search over *collection* embeddings.

        Args:
            collection: Name of the collection to search.
            query_embedding: The query vector.
            top_k: Maximum results.
            threshold: Minimum cosine-similarity score.
            include_documents: If True, include the full document in results.

        Returns:
            List of dicts with ``doc_id``, ``score``, and optionally ``document``.
        """
        idx = VectorIndex(collection, self._storage)
        hits = idx.search(query_embedding, top_k=top_k, threshold=threshold)

        results: list[dict[str, Any]] = []
        coll = self.get_collection(collection) if include_documents else None
        for hit in hits:
            entry: dict[str, Any] = {"doc_id": hit.doc_id, "score": hit.score}
            if coll is not None:
                entry["document"] = coll.get(hit.doc_id).to_dict()
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release database resources."""
        self._storage.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Database(path={self._storage._path!r})"
