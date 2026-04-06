"""Tests for the REST API."""

import pytest
from fastapi.testclient import TestClient

from llmdb.api.server import create_app


@pytest.fixture
def client():
    app = create_app(":memory:")
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "LLMDatabase"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def test_create_collection(client):
    r = client.post("/collections", json={"name": "test"})
    assert r.status_code == 201
    assert r.json()["name"] == "test"


def test_list_collections(client):
    client.post("/collections", json={"name": "c1"})
    client.post("/collections", json={"name": "c2"})
    r = client.get("/collections")
    assert r.status_code == 200
    names = {c["name"] for c in r.json()}
    assert {"c1", "c2"} <= names


def test_get_collection(client):
    client.post("/collections", json={"name": "c1"})
    r = client.get("/collections/c1")
    assert r.status_code == 200
    assert r.json()["name"] == "c1"


def test_get_collection_not_found(client):
    r = client.get("/collections/nope")
    assert r.status_code == 404


def test_drop_collection(client):
    client.post("/collections", json={"name": "del_me"})
    r = client.delete("/collections/del_me")
    assert r.status_code == 200
    r2 = client.get("/collections/del_me")
    assert r2.status_code == 404


def test_create_collection_duplicate(client):
    client.post("/collections", json={"name": "dup"})
    r = client.post("/collections", json={"name": "dup"})
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def test_insert_and_get_document(client):
    client.post("/collections", json={"name": "notes"})
    r = client.post(
        "/collections/notes/documents",
        json={"data": {"title": "Hello"}, "tags": ["greeting"]},
    )
    assert r.status_code == 201
    doc_id = r.json()["id"]
    r2 = client.get(f"/collections/notes/documents/{doc_id}")
    assert r2.status_code == 200
    assert r2.json()["data"]["title"] == "Hello"


def test_list_documents(client):
    client.post("/collections", json={"name": "notes"})
    for i in range(3):
        client.post("/collections/notes/documents", json={"data": {"i": i}})
    r = client.get("/collections/notes/documents")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_update_document(client):
    client.post("/collections", json={"name": "notes"})
    r = client.post("/collections/notes/documents", json={"data": {"v": 1}})
    doc_id = r.json()["id"]
    r2 = client.patch(
        f"/collections/notes/documents/{doc_id}", json={"data": {"v": 2}}
    )
    assert r2.status_code == 200
    assert r2.json()["data"]["v"] == 2


def test_delete_document(client):
    client.post("/collections", json={"name": "notes"})
    r = client.post("/collections/notes/documents", json={"data": {"x": 1}})
    doc_id = r.json()["id"]
    r2 = client.delete(f"/collections/notes/documents/{doc_id}")
    assert r2.status_code == 200
    r3 = client.get(f"/collections/notes/documents/{doc_id}")
    assert r3.status_code == 404


def test_query_documents(client):
    client.post("/collections", json={"name": "notes"})
    client.post("/collections/notes/documents", json={"data": {"score": 10}})
    client.post("/collections/notes/documents", json={"data": {"score": 90}})
    r = client.post(
        "/collections/notes/documents/query",
        json={"filter": {"score": {"$gt": 50}}},
    )
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_vector_search_api(client):
    client.post("/collections", json={"name": "vecs", "vector_dim": 2})
    client.post(
        "/collections/vecs/documents",
        json={"data": {"label": "A"}, "embedding": [1.0, 0.0]},
    )
    client.post(
        "/collections/vecs/documents",
        json={"data": {"label": "B"}, "embedding": [0.0, 1.0]},
    )
    r = client.post(
        "/collections/vecs/documents/search",
        json={"embedding": [1.0, 0.0], "top_k": 1},
    )
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert results[0]["document"]["data"]["label"] == "A"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_add_and_list_memory(client):
    r = client.post(
        "/memory",
        json={"content": "Paris is in France", "memory_type": "semantic"},
    )
    assert r.status_code == 201
    r2 = client.get("/memory")
    assert r2.status_code == 200
    assert any(m["content"] == "Paris is in France" for m in r2.json())


def test_get_memory(client):
    r = client.post("/memory", json={"content": "test memory"})
    mem_id = r.json()["id"]
    r2 = client.get(f"/memory/{mem_id}")
    assert r2.status_code == 200
    assert r2.json()["content"] == "test memory"


def test_delete_memory(client):
    r = client.post("/memory", json={"content": "delete me"})
    mem_id = r.json()["id"]
    r2 = client.delete(f"/memory/{mem_id}")
    assert r2.status_code == 200
    r3 = client.get(f"/memory/{mem_id}")
    assert r3.status_code == 404


def test_memory_semantic_search(client):
    client.post("/memory", json={"content": "A", "embedding": [1.0, 0.0]})
    client.post("/memory", json={"content": "B", "embedding": [0.0, 1.0]})
    r = client.post("/memory/search", json={"embedding": [1.0, 0.0], "top_k": 1})
    assert r.status_code == 200
    assert r.json()[0]["memory"]["content"] == "A"


def test_memory_count(client):
    r = client.get("/memory/count")
    assert r.status_code == 200
    initial = r.json()["count"]
    client.post("/memory", json={"content": "x"})
    r2 = client.get("/memory/count")
    assert r2.json()["count"] == initial + 1
