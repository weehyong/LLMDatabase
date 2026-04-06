"""Tests for the StorageEngine."""

import pytest
from llmdb.core.storage import StorageEngine


def test_encode_decode_roundtrip():
    obj = {"key": [1, 2, 3], "nested": {"a": True}}
    assert StorageEngine.decode(StorageEngine.encode(obj)) == obj


def test_in_memory_schema_created():
    engine = StorageEngine()
    # collections table must exist
    row = engine.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='collections'"
    ).fetchone()
    assert row is not None


def test_execute_commit():
    engine = StorageEngine()
    engine.execute(
        "INSERT INTO collections (name, metadata) VALUES (?, ?)",
        ("test_col", "{}"),
        commit=True,
    )
    row = engine.execute(
        "SELECT name FROM collections WHERE name = 'test_col'"
    ).fetchone()
    assert row is not None
    assert row["name"] == "test_col"
