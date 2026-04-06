"""Memory routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from llmdb.api.models import (
    AddMemoryRequest,
    CountResponse,
    MemoryOut,
    MemorySearchRequest,
    MemorySearchResult,
    MessageResponse,
)
from llmdb.exceptions import DocumentNotFoundError
from llmdb.memory.store import MemoryType

router = APIRouter(prefix="/memory", tags=["Memory"])


def _db():
    from llmdb.api.server import _app_db
    return _app_db()


def _mem_out(mem) -> MemoryOut:
    d = mem.to_dict()
    d.pop("embedding", None)  # don't return raw vectors in listings
    return MemoryOut(**d)


@router.post("", response_model=MemoryOut, status_code=status.HTTP_201_CREATED)
def add_memory(req: AddMemoryRequest):
    mem = _db().memory.add(
        req.content,
        memory_type=req.memory_type,
        metadata=req.metadata,
        importance=req.importance,
        tags=req.tags,
        embedding=req.embedding,
    )
    return _mem_out(mem)


@router.get("", response_model=list[MemoryOut])
def list_memories(
    memory_type: str | None = Query(None),
    tags: list[str] = Query(default=[]),
    min_importance: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=10_000),
    offset: int = Query(0, ge=0),
):
    mems = _db().memory.list(
        memory_type=memory_type,
        tags=tags or None,
        min_importance=min_importance,
        limit=limit,
        offset=offset,
    )
    return [_mem_out(m) for m in mems]


@router.post("/search", response_model=list[MemorySearchResult])
def search_memories(req: MemorySearchRequest):
    results = _db().memory.search(
        req.embedding,
        top_k=req.top_k,
        threshold=req.threshold,
        memory_type=req.memory_type,
    )
    return [
        MemorySearchResult(memory=_mem_out(mem), score=score)
        for mem, score in results
    ]


@router.get("/count", response_model=CountResponse)
def count_memories(memory_type: str | None = Query(None)):
    return CountResponse(count=_db().memory.count(memory_type))


@router.get("/{memory_id}", response_model=MemoryOut)
def get_memory(memory_id: str):
    try:
        mem = _db().memory.get(memory_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _mem_out(mem)


@router.delete("/{memory_id}", response_model=MessageResponse)
def delete_memory(memory_id: str):
    try:
        _db().memory.delete(memory_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return MessageResponse(message=f"Memory '{memory_id}' deleted.")
