"""Core storage engine — SQLite-backed persistence layer for LLMDatabase."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class StorageEngine:
    """Thread-safe SQLite storage engine.

    All raw persistence (collections metadata, documents, vectors, memories)
    goes through this class so the rest of the system stays storage-agnostic.
    """

    # DDL for all tables --------------------------------------------------
    _SCHEMA = """
    PRAGMA journal_mode=WAL;
    PRAGMA foreign_keys=ON;

    CREATE TABLE IF NOT EXISTS collections (
        name        TEXT PRIMARY KEY,
        metadata    TEXT NOT NULL DEFAULT '{}',
        vector_dim  INTEGER,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS documents (
        id          TEXT NOT NULL,
        collection  TEXT NOT NULL,
        data        TEXT NOT NULL,
        tags        TEXT NOT NULL DEFAULT '[]',
        score       REAL NOT NULL DEFAULT 0.0,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (id, collection),
        FOREIGN KEY (collection) REFERENCES collections(name) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS vectors (
        doc_id      TEXT NOT NULL,
        collection  TEXT NOT NULL,
        embedding   TEXT NOT NULL,
        PRIMARY KEY (doc_id, collection),
        FOREIGN KEY (doc_id, collection) REFERENCES documents(id, collection)
            ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS memories (
        id          TEXT PRIMARY KEY,
        memory_type TEXT NOT NULL,
        content     TEXT NOT NULL,
        metadata    TEXT NOT NULL DEFAULT '{}',
        importance  REAL NOT NULL DEFAULT 0.5,
        tags        TEXT NOT NULL DEFAULT '[]',
        embedding   TEXT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        accessed_at TEXT NOT NULL DEFAULT (datetime('now')),
        access_count INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
    CREATE INDEX IF NOT EXISTS idx_documents_tags       ON documents(tags);
    CREATE INDEX IF NOT EXISTS idx_memories_type        ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_memories_importance  ON memories(importance DESC);
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._local = threading.local()
        # Initialise schema on the calling thread's connection
        self._execute_script(self._SCHEMA)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection (created on first access).

        The schema is applied on every newly created connection so that
        threadpool workers (e.g. FastAPI's run_in_threadpool) always have
        a fully initialised connection.
        """
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(self._SCHEMA)
            conn.commit()
            self._local.conn = conn
        return self._local.conn

    def _execute_script(self, script: str) -> None:
        conn = self._conn()
        conn.executescript(script)
        conn.commit()

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def execute(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        *,
        commit: bool = False,
    ) -> sqlite3.Cursor:
        cursor = self._conn().execute(sql, params)
        if commit:
            self._conn().commit()
        return cursor

    def executemany(
        self,
        sql: str,
        seq: list[tuple[Any, ...]],
        *,
        commit: bool = False,
    ) -> None:
        self._conn().executemany(sql, seq)
        if commit:
            self._conn().commit()

    def commit(self) -> None:
        self._conn().commit()

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    # ------------------------------------------------------------------
    # JSON helpers (stored as TEXT in SQLite)
    # ------------------------------------------------------------------

    @staticmethod
    def encode(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    @staticmethod
    def decode(text: str) -> Any:
        return json.loads(text)
