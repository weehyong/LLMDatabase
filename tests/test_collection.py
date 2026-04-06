"""Tests for Collection and Document operations."""

import pytest
from llmdb import Database, CollectionExistsError, DocumentNotFoundError
from llmdb.exceptions import VectorDimensionError


@pytest.fixture
def db():
    with Database() as d:
        yield d


def test_create_and_get_collection(db):
    coll = db.create_collection("things")
    assert coll.name == "things"
    assert db.collection_exists("things")


def test_create_collection_exist_ok(db):
    db.create_collection("things")
    # Should not raise
    coll2 = db.create_collection("things", exist_ok=True)
    assert coll2.name == "things"


def test_create_collection_duplicate_raises(db):
    db.create_collection("things")
    with pytest.raises(CollectionExistsError):
        db.create_collection("things")


def test_drop_collection(db):
    db.create_collection("things")
    db.drop_collection("things")
    assert not db.collection_exists("things")


def test_list_collections(db):
    db.create_collection("a")
    db.create_collection("b")
    names = {c["name"] for c in db.list_collections()}
    assert {"a", "b"} <= names


def test_insert_and_get_document(db):
    db.create_collection("notes")
    doc = db.insert("notes", {"title": "Hello", "body": "World"})
    assert doc.id
    fetched = db.get("notes", doc.id)
    assert fetched.data["title"] == "Hello"


def test_insert_with_tags_and_score(db):
    db.create_collection("notes")
    doc = db.insert("notes", {"x": 1}, tags=["important"], score=0.9)
    fetched = db.get("notes", doc.id)
    assert "important" in fetched.tags
    assert fetched.score == pytest.approx(0.9)


def test_update_document(db):
    db.create_collection("notes")
    doc = db.insert("notes", {"v": 1})
    updated = db.update("notes", doc.id, {"v": 2, "extra": "yes"})
    assert updated.data["v"] == 2
    assert updated.data["extra"] == "yes"


def test_delete_document(db):
    db.create_collection("notes")
    doc = db.insert("notes", {"x": 1})
    db.delete("notes", doc.id)
    with pytest.raises(DocumentNotFoundError):
        db.get("notes", doc.id)


def test_list_documents(db):
    coll = db.create_collection("notes")
    for i in range(5):
        coll.insert({"i": i}, tags=["batch"])
    docs = coll.list(limit=10)
    assert len(docs) == 5


def test_list_by_tags(db):
    coll = db.create_collection("notes")
    coll.insert({"x": 1}, tags=["a"])
    coll.insert({"x": 2}, tags=["b"])
    coll.insert({"x": 3}, tags=["a", "b"])
    tagged_a = coll.list(tags=["a"])
    assert all("a" in d.tags for d in tagged_a)
    assert len(tagged_a) == 2


def test_count(db):
    coll = db.create_collection("notes")
    assert coll.count() == 0
    coll.insert({"x": 1})
    coll.insert({"x": 2})
    assert coll.count() == 2


def test_upsert_creates_new(db):
    coll = db.create_collection("notes")
    doc = coll.upsert({"x": 99}, id="fixed-id")
    assert doc.id == "fixed-id"
    assert coll.count() == 1


def test_upsert_replaces_existing(db):
    coll = db.create_collection("notes")
    coll.upsert({"x": 1}, id="fixed-id")
    coll.upsert({"x": 2}, id="fixed-id")
    assert coll.count() == 1
    assert coll.get("fixed-id").data["x"] == 2


def test_vector_storage_and_retrieval(db):
    coll = db.create_collection("vecs", vector_dim=3)
    doc = coll.insert({"label": "A"}, embedding=[1.0, 0.0, 0.0])
    emb = coll.get_embedding(doc.id)
    assert emb == [1.0, 0.0, 0.0]


def test_vector_dimension_error(db):
    coll = db.create_collection("vecs", vector_dim=3)
    doc = coll.insert({"x": 1})
    with pytest.raises(VectorDimensionError):
        coll.set_embedding(doc.id, [1.0, 2.0])  # wrong dim


def test_document_not_found_raises(db):
    db.create_collection("notes")
    with pytest.raises(DocumentNotFoundError):
        db.get("notes", "nonexistent-id")
