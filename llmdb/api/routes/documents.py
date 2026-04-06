"""Documents routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from llmdb.api.models import (
    CountResponse,
    DocumentOut,
    InsertDocumentRequest,
    MessageResponse,
    QueryRequest,
    UpdateDocumentRequest,
    UpsertDocumentRequest,
    VectorSearchRequest,
    VectorSearchResult,
)
from llmdb.exceptions import (
    CollectionNotFoundError,
    DocumentNotFoundError,
    VectorDimensionError,
)

router = APIRouter(prefix="/collections/{collection}/documents", tags=["Documents"])


def _db():
    from llmdb.api.server import _app_db
    return _app_db()


def _doc_out(doc) -> DocumentOut:
    return DocumentOut(**doc.to_dict())


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def insert_document(collection: str, req: InsertDocumentRequest):
    try:
        doc = _db().get_collection(collection).insert(
            req.data,
            id=req.id,
            tags=req.tags,
            score=req.score,
            embedding=req.embedding,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except VectorDimensionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _doc_out(doc)


@router.put("/{doc_id}", response_model=DocumentOut)
def upsert_document(collection: str, doc_id: str, req: UpsertDocumentRequest):
    try:
        doc = _db().get_collection(collection).upsert(
            req.data,
            id=doc_id,
            tags=req.tags,
            score=req.score,
            embedding=req.embedding,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except VectorDimensionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _doc_out(doc)


@router.get("", response_model=list[DocumentOut])
def list_documents(
    collection: str,
    tags: list[str] = Query(default=[]),
    limit: int = Query(100, ge=1, le=10_000),
    offset: int = Query(0, ge=0),
):
    try:
        docs = _db().get_collection(collection).list(
            tags=tags or None, limit=limit, offset=offset
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return [_doc_out(d) for d in docs]


@router.post("/query", response_model=list[DocumentOut])
def query_documents(collection: str, req: QueryRequest):
    try:
        docs = _db().query(
            collection,
            req.filter,
            sort_by=req.sort_by,
            descending=req.descending,
            limit=req.limit,
            offset=req.offset,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return [_doc_out(d) for d in docs]


@router.get("/count", response_model=CountResponse)
def count_documents(collection: str):
    try:
        c = _db().get_collection(collection).count()
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return CountResponse(count=c)


@router.post("/search", response_model=list[VectorSearchResult])
def vector_search(collection: str, req: VectorSearchRequest):
    try:
        results = _db().vector_search(
            collection,
            req.embedding,
            top_k=req.top_k,
            threshold=req.threshold,
            include_documents=req.include_documents,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    out = []
    for r in results:
        doc_out = None
        if r.get("document"):
            from llmdb.core.document import Document
            doc_out = DocumentOut(**r["document"])
        out.append(VectorSearchResult(doc_id=r["doc_id"], score=r["score"], document=doc_out))
    return out


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(collection: str, doc_id: str):
    try:
        doc = _db().get(collection, doc_id)
    except (CollectionNotFoundError, DocumentNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _doc_out(doc)


@router.patch("/{doc_id}", response_model=DocumentOut)
def update_document(collection: str, doc_id: str, req: UpdateDocumentRequest):
    try:
        doc = _db().update(collection, doc_id, req.data)
    except (CollectionNotFoundError, DocumentNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _doc_out(doc)


@router.delete("/{doc_id}", response_model=MessageResponse)
def delete_document(collection: str, doc_id: str):
    try:
        _db().delete(collection, doc_id)
    except (CollectionNotFoundError, DocumentNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return MessageResponse(message=f"Document '{doc_id}' deleted.")
