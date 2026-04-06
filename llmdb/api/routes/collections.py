"""Collections routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from llmdb.api.models import (
    CollectionInfo,
    CreateCollectionRequest,
    MessageResponse,
)
from llmdb.exceptions import CollectionExistsError, CollectionNotFoundError

router = APIRouter(prefix="/collections", tags=["Collections"])


def _db():
    """Dependency: retrieve the Database instance attached to the app state."""
    from llmdb.api.server import _app_db  # local import to avoid cycles
    return _app_db()


@router.post("", response_model=CollectionInfo, status_code=status.HTTP_201_CREATED)
def create_collection(req: CreateCollectionRequest):
    db = _db()
    try:
        coll = db.create_collection(
            req.name,
            vector_dim=req.vector_dim,
            metadata=req.metadata,
            exist_ok=req.exist_ok,
        )
    except CollectionExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    row = next(c for c in db.list_collections() if c["name"] == coll.name)
    return CollectionInfo(**row)


@router.get("", response_model=list[CollectionInfo])
def list_collections():
    return [CollectionInfo(**c) for c in _db().list_collections()]


@router.get("/{name}", response_model=CollectionInfo)
def get_collection(name: str):
    db = _db()
    try:
        coll = db.get_collection(name)
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    row = next(c for c in db.list_collections() if c["name"] == name)
    return CollectionInfo(**row)


@router.delete("/{name}", response_model=MessageResponse)
def drop_collection(name: str):
    try:
        _db().drop_collection(name)
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return MessageResponse(message=f"Collection '{name}' deleted.")
