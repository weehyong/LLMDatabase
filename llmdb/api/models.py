"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class CreateCollectionRequest(BaseModel):
    name: str
    vector_dim: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    exist_ok: bool = False


class CollectionInfo(BaseModel):
    name: str
    vector_dim: int | None
    metadata: dict[str, Any]
    created_at: str


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class InsertDocumentRequest(BaseModel):
    data: dict[str, Any]
    id: str | None = None
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    embedding: list[float] | None = None


class UpsertDocumentRequest(BaseModel):
    data: dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    embedding: list[float] | None = None


class UpdateDocumentRequest(BaseModel):
    data: dict[str, Any]


class DocumentOut(BaseModel):
    id: str
    collection: str
    data: dict[str, Any]
    tags: list[str]
    score: float
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    filter: dict[str, Any] | None = None
    sort_by: str | None = None
    descending: bool = True
    limit: int = Field(100, ge=1, le=10_000)
    offset: int = Field(0, ge=0)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class VectorSearchRequest(BaseModel):
    embedding: list[float]
    top_k: int = Field(10, ge=1, le=1_000)
    threshold: float = -1.0
    include_documents: bool = True


class VectorSearchResult(BaseModel):
    doc_id: str
    score: float
    document: DocumentOut | None = None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class AddMemoryRequest(BaseModel):
    content: str
    memory_type: str = "episodic"
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class MemoryOut(BaseModel):
    id: str
    content: str
    memory_type: str
    metadata: dict[str, Any]
    importance: float
    tags: list[str]
    created_at: str
    accessed_at: str
    access_count: int


class MemorySearchRequest(BaseModel):
    embedding: list[float]
    top_k: int = Field(10, ge=1)
    threshold: float = 0.0
    memory_type: str | None = None


class MemorySearchResult(BaseModel):
    memory: MemoryOut
    score: float


# ---------------------------------------------------------------------------
# Generic responses
# ---------------------------------------------------------------------------


class MessageResponse(BaseModel):
    message: str


class CountResponse(BaseModel):
    count: int
