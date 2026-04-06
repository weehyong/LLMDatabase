"""Tests for the VectorIndex."""

import math
import pytest
from llmdb import Database


@pytest.fixture
def db():
    with Database() as d:
        yield d


def test_vector_search_basic(db):
    coll = db.create_collection("vecs", vector_dim=3)
    doc_a = coll.insert({"label": "X"}, embedding=[1.0, 0.0, 0.0])
    doc_b = coll.insert({"label": "Y"}, embedding=[0.0, 1.0, 0.0])
    doc_c = coll.insert({"label": "Z"}, embedding=[0.9, 0.1, 0.0])

    results = db.vector_search("vecs", [1.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3
    # doc_a should be the top result (perfect match)
    assert results[0]["doc_id"] == doc_a.id
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)


def test_vector_search_top_k_limits(db):
    coll = db.create_collection("vecs")
    for i in range(10):
        v = [float(i == j) for j in range(10)]
        coll.insert({"i": i}, embedding=v)
    results = db.vector_search("vecs", [1.0] + [0.0] * 9, top_k=3)
    assert len(results) == 3


def test_vector_search_threshold(db):
    coll = db.create_collection("vecs")
    coll.insert({"a": 1}, embedding=[1.0, 0.0])
    coll.insert({"b": 2}, embedding=[0.0, 1.0])  # perpendicular → sim=0
    results = db.vector_search("vecs", [1.0, 0.0], top_k=10, threshold=0.5)
    # Only the first doc should survive threshold
    assert len(results) == 1
    assert results[0]["score"] >= 0.5


def test_vector_search_includes_document(db):
    coll = db.create_collection("vecs")
    doc = coll.insert({"hello": "world"}, embedding=[1.0, 0.0])
    results = db.vector_search("vecs", [1.0, 0.0], include_documents=True)
    assert results[0]["document"]["data"]["hello"] == "world"


def test_vector_search_excludes_document(db):
    coll = db.create_collection("vecs")
    coll.insert({"hello": "world"}, embedding=[1.0, 0.0])
    results = db.vector_search("vecs", [1.0, 0.0], include_documents=False)
    assert "document" not in results[0] or results[0].get("document") is None


def test_cosine_similarity_zero_vector():
    from llmdb.vector.index import _cosine_similarity
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_identical():
    from llmdb.vector.index import _cosine_similarity
    v = [0.3, 0.4]
    assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal():
    from llmdb.vector.index import _cosine_similarity
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)
