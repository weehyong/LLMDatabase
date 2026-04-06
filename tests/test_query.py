"""Tests for the QueryEngine."""

import pytest
from llmdb import Database


@pytest.fixture
def populated_db():
    with Database() as db:
        coll = db.create_collection("articles")
        coll.insert({"title": "Intro to AI",    "views": 500, "author": "alice", "draft": False})
        coll.insert({"title": "Deep Learning",  "views": 200, "author": "bob",   "draft": False})
        coll.insert({"title": "RL Survey",       "views": 800, "author": "alice", "draft": True})
        coll.insert({"title": "NLP Basics",      "views": 150, "author": "carol", "draft": False})
        coll.insert({"title": "Vector DBs",      "views": 300, "author": "alice", "draft": False})
        yield db


def test_query_all(populated_db):
    docs = populated_db.query("articles")
    assert len(docs) == 5


def test_query_eq(populated_db):
    docs = populated_db.query("articles", {"author": {"$eq": "alice"}})
    assert len(docs) == 3
    assert all(d.data["author"] == "alice" for d in docs)


def test_query_eq_shorthand(populated_db):
    docs = populated_db.query("articles", {"author": "alice"})
    assert len(docs) == 3


def test_query_ne(populated_db):
    docs = populated_db.query("articles", {"author": {"$ne": "alice"}})
    assert len(docs) == 2


def test_query_gt(populated_db):
    docs = populated_db.query("articles", {"views": {"$gt": 300}})
    assert all(d.data["views"] > 300 for d in docs)


def test_query_lte(populated_db):
    docs = populated_db.query("articles", {"views": {"$lte": 200}})
    assert all(d.data["views"] <= 200 for d in docs)


def test_query_in(populated_db):
    docs = populated_db.query("articles", {"author": {"$in": ["alice", "carol"]}})
    assert len(docs) == 4


def test_query_contains(populated_db):
    docs = populated_db.query("articles", {"title": {"$contains": "deep"}})
    assert len(docs) == 1
    assert docs[0].data["title"] == "Deep Learning"


def test_query_exists_true(populated_db):
    docs = populated_db.query("articles", {"draft": {"$exists": True}})
    assert len(docs) == 5  # all have 'draft'


def test_query_and(populated_db):
    docs = populated_db.query(
        "articles",
        {"$and": [{"author": "alice"}, {"draft": False}]},
    )
    assert len(docs) == 2
    assert all(d.data["draft"] is False for d in docs)


def test_query_or(populated_db):
    docs = populated_db.query(
        "articles",
        {"$or": [{"author": "bob"}, {"author": "carol"}]},
    )
    assert len(docs) == 2


def test_query_not(populated_db):
    docs = populated_db.query("articles", {"$not": {"draft": True}})
    assert all(d.data["draft"] is not True for d in docs)


def test_query_sort_by(populated_db):
    docs = populated_db.query("articles", sort_by="views", descending=True)
    views = [d.data["views"] for d in docs]
    assert views == sorted(views, reverse=True)


def test_query_sort_ascending(populated_db):
    docs = populated_db.query("articles", sort_by="views", descending=False)
    views = [d.data["views"] for d in docs]
    assert views == sorted(views)


def test_query_limit_offset(populated_db):
    all_docs = populated_db.query("articles", sort_by="views", descending=True)
    page1 = populated_db.query("articles", sort_by="views", descending=True, limit=2, offset=0)
    page2 = populated_db.query("articles", sort_by="views", descending=True, limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0].id != page2[0].id
