"""LLMDatabase — a database management system for agentic applications."""

from llmdb.engine import Database
from llmdb.exceptions import (
    CollectionExistsError,
    CollectionNotFoundError,
    DocumentNotFoundError,
    LLMDBError,
    QueryError,
    VectorDimensionError,
)
from llmdb.memory.store import Memory, MemoryStore, MemoryType

__all__ = [
    "Database",
    "Memory",
    "MemoryStore",
    "MemoryType",
    # Exceptions
    "LLMDBError",
    "CollectionNotFoundError",
    "CollectionExistsError",
    "DocumentNotFoundError",
    "VectorDimensionError",
    "QueryError",
]

__version__ = "0.1.0"
