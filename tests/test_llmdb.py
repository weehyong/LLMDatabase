"""
Tests for LLMDatabase.

Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from llmdb import Database
from llmdb.core.memory import MemoryManager, SlabPool, MemoryArena
from llmdb.core.storage import StorageManager, Page, PAGE_SIZE
from llmdb.core.buffer import BufferPoolManager, PageId
from llmdb.core.actors import (
    DbObject, DbObjectType, ColumnDef, ColType,
    ColumnSetComponent, IndexComponent, SchemaFactory,
)
from llmdb.core.events import EventBus, QueryReceivedEvent, QueryResultEvent
from llmdb.core.process import ProcessManager, CallbackProcess, DelayProcess
from llmdb.query.sql import parse_sql, SelectStmt, InsertStmt, CreateTableStmt
from llmdb.query.natural import RuleBasedTranslator, build_schema_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def db(tmp_dir):
    d = Database(tmp_dir)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# Memory Manager Tests
# ---------------------------------------------------------------------------

class TestMemoryManager:
    def test_slab_pool_alloc_free(self):
        pool = SlabPool(slab_size=64, capacity=4)
        assert pool.free_count == 4
        b = pool.allocate("test")
        assert pool.free_count == 3
        pool.free(b)
        assert pool.free_count == 4

    def test_slab_pool_exhaustion(self):
        pool = SlabPool(slab_size=64, capacity=2)
        b1 = pool.allocate("a")
        b2 = pool.allocate("b")
        with pytest.raises(MemoryError):
            pool.allocate("c")
        pool.free(b1)
        b3 = pool.allocate("c")  # should succeed now
        pool.free(b2)
        pool.free(b3)

    def test_memory_arena(self):
        arena = MemoryArena(capacity=1024)
        view = arena.allocate(100)
        assert len(view) == 100
        assert arena.used_bytes == 100
        arena.reset()
        assert arena.used_bytes == 0

    def test_memory_arena_overflow(self):
        arena = MemoryArena(capacity=10)
        with pytest.raises(MemoryError):
            arena.allocate(11)

    def test_memory_manager_report(self):
        mm = MemoryManager(page_pool_capacity=8, row_pool_capacity=16)
        report = mm.report()
        assert report["page_pool"]["capacity"] == 8
        assert report["row_pool"]["capacity"] == 16

    def test_remaining_page_memory(self):
        mm = MemoryManager(page_size=8192, page_pool_capacity=4)
        initial = mm.remaining_page_memory()
        block = mm.alloc_page_frame("test")
        assert mm.remaining_page_memory() == initial - 8192
        mm.free_page_frame(block)
        assert mm.remaining_page_memory() == initial


# ---------------------------------------------------------------------------
# Storage Manager Tests
# ---------------------------------------------------------------------------

class TestStorageManager:
    def test_create_and_list_relation(self, tmp_dir):
        sm = StorageManager(tmp_dir)
        sm.create_relation("users")
        assert "users" in sm.list_relations()
        sm.close()

    def test_read_write_page(self, tmp_dir):
        sm = StorageManager(tmp_dir)
        sm.create_relation("t1")
        page = sm.read_page("t1", 0)
        assert len(page) == PAGE_SIZE
        sm.close()

    def test_alloc_page(self, tmp_dir):
        sm = StorageManager(tmp_dir)
        sm.create_relation("t1")
        new_id = sm.alloc_page("t1")
        assert new_id == 1
        assert sm.page_count("t1") == 2
        sm.close()

    def test_drop_relation(self, tmp_dir):
        sm = StorageManager(tmp_dir)
        sm.create_relation("drop_me")
        sm.drop_relation("drop_me")
        assert "drop_me" not in sm.list_relations()
        sm.close()


# ---------------------------------------------------------------------------
# Buffer Pool Tests
# ---------------------------------------------------------------------------

class TestBufferPool:
    def test_fetch_page_cache_miss_and_hit(self, tmp_dir):
        mm = MemoryManager(page_pool_capacity=8)
        sm = StorageManager(tmp_dir)
        sm.create_relation("buf_test")
        pool = BufferPoolManager(mm, sm, num_frames=4)

        pid = PageId("buf_test", 0)
        with pool.fetch_page(pid) as pp:
            assert pp.page is not None

        stats = pool.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        with pool.fetch_page(pid) as pp2:
            pass  # should be a hit
        stats2 = pool.stats()
        assert stats2["hits"] == 1
        sm.close()

    def test_remaining_memory_decreases(self, tmp_dir):
        mm = MemoryManager(page_pool_capacity=8)
        sm = StorageManager(tmp_dir)
        sm.create_relation("rm_test")
        pool = BufferPoolManager(mm, sm, num_frames=4)

        before = pool.remaining_memory
        pid = PageId("rm_test", 0)
        with pool.fetch_page(pid) as pp:
            pp.mark_dirty()
            during = pool.remaining_memory

        # After unpin the frame is still used (not evicted) but dirty
        assert during <= before
        sm.close()


# ---------------------------------------------------------------------------
# Actor / Component Tests
# ---------------------------------------------------------------------------

class TestActors:
    def test_schema_factory_creates_table(self):
        schema = {
            "name": "products",
            "type": "table",
            "columns": [
                {"name": "id",    "type": "INTEGER", "primary_key": True},
                {"name": "price", "type": "REAL",    "nullable": False},
                {"name": "name",  "type": "TEXT"},
            ],
        }
        obj = SchemaFactory.create(schema)
        assert obj.name == "products"
        assert obj.obj_type == DbObjectType.TABLE

        col_comp = obj.get_component_typed(ColumnSetComponent)
        assert col_comp is not None
        assert len(col_comp.columns) == 3

    def test_index_component_lookup(self):
        idx = IndexComponent("idx_test", ["id"], unique=True)
        idx.insert(1, 100)
        idx.insert(2, 200)
        idx.insert(3, 300)
        assert idx.lookup(2) == [200]
        assert idx.lookup(99) == []

    def test_index_range_scan(self):
        idx = IndexComponent("idx_range", ["score"])
        for i in range(10):
            idx.insert(i, i * 10)
        results = list(idx.range_scan(3, 6))
        assert [k for k, _ in results] == [3, 4, 5, 6]

    def test_attach_detach_component(self):
        obj = DbObject("t", DbObjectType.TABLE)
        col_comp = ColumnSetComponent([ColumnDef("id", ColType.INTEGER)])
        obj.add_component(col_comp)
        assert obj.has_component("ColumnSet")
        obj.remove_component("ColumnSet")
        assert not obj.has_component("ColumnSet")


# ---------------------------------------------------------------------------
# Event Bus Tests
# ---------------------------------------------------------------------------

class TestEventBus:
    def test_post_fires_listener(self):
        bus = EventBus()
        received = []
        bus.subscribe("query.received", lambda e: received.append(e))
        evt = QueryReceivedEvent(query_text="SELECT 1", session_id="s1")
        bus.post(evt)
        assert len(received) == 1
        assert received[0].query_text == "SELECT 1"

    def test_wildcard_listener(self):
        bus = EventBus()
        all_events = []
        bus.subscribe("*", lambda e: all_events.append(e))
        bus.post(QueryReceivedEvent(query_text="x", session_id="s"))
        bus.post(QueryResultEvent(session_id="s", rows=[], row_count=0, affected_rows=0, elapsed_ms=1))
        assert len(all_events) == 2

    def test_queued_events(self):
        bus = EventBus()
        received = []
        bus.subscribe("query.received", lambda e: received.append(e))
        bus.queue(QueryReceivedEvent(query_text="Q", session_id="s"))
        assert len(received) == 0
        bus.process_queued()
        assert len(received) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        fn = lambda e: received.append(e)
        bus.subscribe("query.received", fn)
        bus.unsubscribe("query.received", fn)
        bus.post(QueryReceivedEvent(query_text="x", session_id="s"))
        assert len(received) == 0


# ---------------------------------------------------------------------------
# Process Manager Tests
# ---------------------------------------------------------------------------

class TestProcessManager:
    def test_callback_process_runs(self):
        pm = ProcessManager(tick_interval=0.01)
        pm.start()
        import time
        ran = []
        proc = pm.attach(CallbackProcess(lambda: ran.append(1)))
        time.sleep(0.2)
        pm.stop()
        assert len(ran) == 1

    def test_process_chaining(self):
        import time
        pm = ProcessManager(tick_interval=0.01)
        pm.start()
        order = []
        first = CallbackProcess(lambda: order.append("first"))
        second = CallbackProcess(lambda: order.append("second"))
        first.attach_child(second)
        pm.attach(first)
        time.sleep(0.3)
        pm.stop()
        assert order == ["first", "second"]


# ---------------------------------------------------------------------------
# SQL Parser Tests
# ---------------------------------------------------------------------------

class TestSQLParser:
    def test_select_star(self):
        ast = parse_sql("SELECT * FROM users")
        assert isinstance(ast, SelectStmt)
        assert ast.columns == ["*"]
        assert ast.from_table == "users"

    def test_select_with_where(self):
        ast = parse_sql("SELECT name, email FROM users WHERE id = 1")
        assert isinstance(ast, SelectStmt)
        assert ast.where is not None
        assert "id" in ast.where.text

    def test_insert(self):
        ast = parse_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        assert isinstance(ast, InsertStmt)
        assert ast.table_name == "users"
        assert ast.values == [[1, "Alice"]]

    def test_create_table(self):
        ast = parse_sql(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price REAL)"
        )
        assert isinstance(ast, CreateTableStmt)
        assert ast.table_name == "products"
        assert len(ast.columns) == 3
        assert ast.columns[0].primary_key is True

    def test_select_limit_offset(self):
        ast = parse_sql("SELECT * FROM t LIMIT 10 OFFSET 5")
        assert ast.limit == 10
        assert ast.offset == 5

    def test_select_order_by(self):
        ast = parse_sql("SELECT * FROM t ORDER BY score DESC, name ASC")
        assert ast.order_by == [("score", "DESC"), ("name", "ASC")]


# ---------------------------------------------------------------------------
# Natural Language Translator Tests
# ---------------------------------------------------------------------------

class TestNaturalLanguageTranslator:
    def _ctx(self):
        return {
            "tables": {
                "users": {
                    "columns": [
                        {"name": "id",    "type": "INTEGER", "primary_key": True,  "nullable": False},
                        {"name": "name",  "type": "TEXT",    "primary_key": False, "nullable": True},
                        {"name": "email", "type": "TEXT",    "primary_key": False, "nullable": True},
                        {"name": "score", "type": "REAL",    "primary_key": False, "nullable": True},
                    ]
                },
                "orders": {
                    "columns": [
                        {"name": "id",     "type": "INTEGER", "primary_key": True,  "nullable": False},
                        {"name": "amount", "type": "REAL",    "primary_key": False, "nullable": True},
                    ]
                },
            }
        }

    def test_count_query(self):
        t = RuleBasedTranslator()
        r = t.translate("How many users are there?", self._ctx())
        assert "COUNT" in r.sql.upper()
        assert "users" in r.sql.lower()

    def test_select_all(self):
        t = RuleBasedTranslator()
        r = t.translate("Show me all users", self._ctx())
        assert "users" in r.sql.lower()
        assert "SELECT" in r.sql.upper()

    def test_top_n(self):
        t = RuleBasedTranslator()
        r = t.translate("Top 5 users by score", self._ctx())
        assert "LIMIT 5" in r.sql
        assert "score" in r.sql.lower()

    def test_filter(self):
        t = RuleBasedTranslator()
        r = t.translate("Users with score greater than 80", self._ctx())
        assert "WHERE" in r.sql.upper()
        assert "users" in r.sql.lower()

    def test_sum_aggregate(self):
        t = RuleBasedTranslator()
        r = t.translate("What is the total of amount in orders?", self._ctx())
        assert "SUM" in r.sql.upper()
        assert "orders" in r.sql.lower()


# ---------------------------------------------------------------------------
# Integration Tests (full stack through Database)
# ---------------------------------------------------------------------------

class TestDatabaseIntegration:
    def test_create_insert_select(self, db):
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO t (id, name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO t (id, name) VALUES (2, 'Bob')")
        result = db.execute("SELECT * FROM t")
        assert result.ok
        assert result.row_count == 2

    def test_where_filter(self, db):
        db.execute("CREATE TABLE scores (id INTEGER PRIMARY KEY, score REAL)")
        db.execute("INSERT INTO scores (id, score) VALUES (1, 90.0)")
        db.execute("INSERT INTO scores (id, score) VALUES (2, 60.0)")
        db.execute("INSERT INTO scores (id, score) VALUES (3, 75.0)")
        result = db.execute("SELECT * FROM scores WHERE score > 70")
        assert result.ok
        assert result.row_count == 2

    def test_update(self, db):
        db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)")
        db.execute("INSERT INTO items (id, value) VALUES (1, 10)")
        db.execute("UPDATE items SET value = 99 WHERE id = 1")
        result = db.execute("SELECT * FROM items WHERE id = 1")
        assert result.rows[0]["value"] == 99

    def test_delete(self, db):
        db.execute("CREATE TABLE deletable (id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO deletable (id, name) VALUES (1, 'x')")
        db.execute("INSERT INTO deletable (id, name) VALUES (2, 'y')")
        db.execute("DELETE FROM deletable WHERE id = 1")
        result = db.execute("SELECT * FROM deletable")
        assert result.row_count == 1

    def test_limit_offset(self, db):
        db.execute("CREATE TABLE nums (id INTEGER PRIMARY KEY)")
        for i in range(10):
            db.execute(f"INSERT INTO nums (id) VALUES ({i})")
        result = db.execute("SELECT * FROM nums LIMIT 3 OFFSET 2")
        assert result.row_count == 3

    def test_order_by(self, db):
        db.execute("CREATE TABLE rank (id INTEGER PRIMARY KEY, score REAL)")
        db.execute("INSERT INTO rank (id, score) VALUES (1, 30.0)")
        db.execute("INSERT INTO rank (id, score) VALUES (2, 90.0)")
        db.execute("INSERT INTO rank (id, score) VALUES (3, 60.0)")
        result = db.execute("SELECT * FROM rank ORDER BY score DESC")
        assert result.ok
        scores = [r["score"] for r in result.rows]
        assert scores == sorted(scores, reverse=True)

    def test_drop_table_if_exists(self, db):
        db.execute("CREATE TABLE temp (id INTEGER)")
        db.execute("DROP TABLE temp")
        result = db.execute("DROP TABLE IF EXISTS temp")  # should not error
        assert result.ok

    def test_natural_language_count(self, db):
        db.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, price REAL)")
        db.execute("INSERT INTO products (id, price) VALUES (1, 9.99)")
        db.execute("INSERT INTO products (id, price) VALUES (2, 4.99)")
        result = db.ask("How many products are there?")
        assert result.ok

    def test_status_report(self, db):
        status = db.status()
        assert "memory" in status
        assert "buffer_pool" in status
        assert "remaining_page_memory_bytes" in status["memory"]
        assert status["memory"]["remaining_page_memory_bytes"] > 0

    def test_remaining_memory(self, db):
        mem = db.remaining_memory()
        assert mem > 0

    def test_context_manager(self, tmp_dir):
        with Database(tmp_dir / "ctx_db") as db2:
            db2.execute("CREATE TABLE x (id INTEGER)")
            db2.execute("INSERT INTO x (id) VALUES (42)")
            r = db2.execute("SELECT * FROM x")
            assert r.row_count == 1
