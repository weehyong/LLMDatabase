"""Tests for the MemoryStore."""

import pytest
from llmdb import Database, MemoryType
from llmdb.exceptions import DocumentNotFoundError


@pytest.fixture
def db():
    with Database() as d:
        yield d


def test_add_and_get_memory(db):
    mem = db.memory.add("Paris is the capital of France.", memory_type="semantic")
    fetched = db.memory.get(mem.id)
    assert fetched.content == "Paris is the capital of France."
    assert fetched.memory_type == MemoryType.SEMANTIC


def test_access_count_increments(db):
    mem = db.memory.add("Hello world")
    assert mem.access_count == 0
    db.memory.get(mem.id)
    db.memory.get(mem.id)
    refreshed = db.memory.get(mem.id)
    # after 3 gets the count should be >= 2 (the third get itself increments)
    assert refreshed.access_count >= 2


def test_memory_importance_clamped(db):
    mem = db.memory.add("test", importance=99.0)
    assert mem.importance == 1.0
    mem2 = db.memory.add("test2", importance=-5.0)
    assert mem2.importance == 0.0


def test_list_memories_by_type(db):
    db.memory.add("event A", memory_type="episodic")
    db.memory.add("fact B", memory_type="semantic")
    db.memory.add("skill C", memory_type="procedural")
    episodic = db.memory.list(memory_type="episodic")
    assert all(m.memory_type == MemoryType.EPISODIC for m in episodic)
    assert len(episodic) == 1


def test_list_by_min_importance(db):
    db.memory.add("low", importance=0.1)
    db.memory.add("high", importance=0.9)
    result = db.memory.list(min_importance=0.5)
    assert all(m.importance >= 0.5 for m in result)


def test_list_by_tags(db):
    db.memory.add("A", tags=["coding"])
    db.memory.add("B", tags=["cooking"])
    db.memory.add("C", tags=["coding", "ai"])
    coding_mems = db.memory.list(tags=["coding"])
    assert all("coding" in m.tags for m in coding_mems)
    assert len(coding_mems) == 2


def test_delete_memory(db):
    mem = db.memory.add("to be deleted")
    db.memory.delete(mem.id)
    with pytest.raises(DocumentNotFoundError):
        db.memory.get(mem.id)


def test_memory_count(db):
    assert db.memory.count() == 0
    db.memory.add("a", memory_type="episodic")
    db.memory.add("b", memory_type="semantic")
    assert db.memory.count() == 2
    assert db.memory.count(memory_type="episodic") == 1


def test_semantic_search(db):
    # Build simple 2D embeddings
    db.memory.add("A", embedding=[1.0, 0.0], importance=0.5)
    db.memory.add("B", embedding=[0.0, 1.0], importance=0.5)
    db.memory.add("C", embedding=[0.9, 0.1], importance=0.5)

    results = db.memory.search([1.0, 0.0], top_k=2)
    ids = [m.id for m, _ in results]
    # The closest to [1,0] should be the ones with x-heavy embeddings
    assert len(results) == 2
    # scores should be descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_by_type(db):
    db.memory.add("ep", memory_type="episodic", embedding=[1.0, 0.0])
    db.memory.add("sem", memory_type="semantic", embedding=[1.0, 0.0])
    results = db.memory.search([1.0, 0.0], top_k=5, memory_type="episodic")
    assert all(m.memory_type == MemoryType.EPISODIC for m, _ in results)
