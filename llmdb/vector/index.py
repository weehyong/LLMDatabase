"""Vector index — cosine-similarity search over document embeddings."""

from __future__ import annotations

import math
from typing import NamedTuple

from llmdb.core.storage import StorageEngine


class SearchResult(NamedTuple):
    doc_id: str
    score: float  # cosine similarity in [−1, 1]; higher is more similar


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (no external deps required)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorIndex:
    """Semantic search over embeddings stored in a Collection.

    All embeddings are loaded from SQLite into memory for each search call.
    For production-scale workloads this can be replaced with a specialised
    ANNS backend (FAISS, Annoy, etc.) behind the same interface.
    """

    def __init__(self, collection: str, storage: StorageEngine) -> None:
        self._collection = collection
        self._storage = storage

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        threshold: float = -1.0,
    ) -> list[SearchResult]:
        """Return up to *top_k* document IDs ranked by cosine similarity.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score; results below this are dropped.

        Returns:
            List of :class:`SearchResult` sorted by score descending.
        """
        rows = self._storage.execute(
            "SELECT doc_id, embedding FROM vectors WHERE collection = ?",
            (self._collection,),
        ).fetchall()

        results: list[SearchResult] = []
        for row in rows:
            emb = self._storage.decode(row["embedding"])
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                results.append(SearchResult(doc_id=row["doc_id"], score=sim))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        """Return the number of indexed vectors for this collection."""
        row = self._storage.execute(
            "SELECT COUNT(*) AS c FROM vectors WHERE collection = ?",
            (self._collection,),
        ).fetchone()
        return row["c"] if row else 0
