"""Custom exceptions for LLMDatabase."""


class LLMDBError(Exception):
    """Base exception for all LLMDatabase errors."""


class CollectionNotFoundError(LLMDBError):
    """Raised when a referenced collection does not exist."""


class CollectionExistsError(LLMDBError):
    """Raised when trying to create a collection that already exists."""


class DocumentNotFoundError(LLMDBError):
    """Raised when a referenced document does not exist."""


class VectorDimensionError(LLMDBError):
    """Raised when vector dimensions do not match the collection's dimension."""


class QueryError(LLMDBError):
    """Raised when a query cannot be parsed or executed."""
