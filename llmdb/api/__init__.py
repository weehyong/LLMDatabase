"""
Public Database API — the entry point for all client code.

Usage
-----
>>> from llmdb import Database
>>> db = Database("./my_db")
>>> db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
>>> db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
>>> result = db.execute("SELECT * FROM users")
>>> result.rows
[{'id': 1, 'name': 'Alice'}]

Natural language (English) queries
-----------------------------------
>>> result = db.ask("Who are all the users?")
>>> result = db.ask("How many users are there?")
>>> result = db.ask("Show me users whose name starts with A")

System status / memory report
------------------------------
>>> db.status()   # returns a dict with memory, buffer, storage stats
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmdb.core.buffer import BufferPoolManager, PageId
from llmdb.core.catalog import Catalog
from llmdb.core.events import (
    EventBus,
    QueryReceivedEvent,
    QueryResultEvent,
    SchemaChangedEvent,
    reset_event_bus,
)
from llmdb.core.memory import MemoryManager
from llmdb.core.process import CallbackProcess, LoopProcess, ProcessManager
from llmdb.core.storage import StorageManager
from llmdb.query.executor import ExecutionResult, Executor
from llmdb.query.natural import (
    NaturalLanguageTranslator,
    RuleBasedTranslator,
    build_schema_context,
    make_translator,
)
from llmdb.query.planner import Planner
from llmdb.query.sql import parse_sql


class QueryResult:
    """Returned by Database.execute() and Database.ask()."""

    def __init__(self, result: ExecutionResult, sql: Optional[str] = None) -> None:
        self._result = result
        self._sql = sql

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return self._result.rows

    @property
    def row_count(self) -> int:
        return self._result.row_count

    @property
    def affected_rows(self) -> int:
        return self._result.affected_rows

    @property
    def elapsed_ms(self) -> float:
        return self._result.elapsed_ms

    @property
    def error(self) -> Optional[str]:
        return self._result.error

    @property
    def ok(self) -> bool:
        return self._result.error is None

    @property
    def sql(self) -> Optional[str]:
        return self._sql

    def __repr__(self) -> str:
        if self.error:
            return f"QueryResult(error={self.error!r})"
        return (
            f"QueryResult(rows={self.row_count}, "
            f"affected={self.affected_rows}, "
            f"elapsed={self.elapsed_ms:.1f}ms)"
        )


class Database:
    """LLMDatabase — an OpenClaw-inspired DBMS for agentic applications.

    Architecture overview
    ---------------------
    ┌─────────────────────────────────────────────────────────────┐
    │  Database (public API)                                       │
    │  ┌────────────────┐  ┌──────────────────────────────────┐  │
    │  │ NL Translator  │  │  SQL Parser → Planner → Executor │  │
    │  └────────────────┘  └──────────────────────────────────┘  │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │  Catalog  (system registry — OpenClaw GameLogic)    │   │
    │  └─────────────────────────────────────────────────────┘   │
    │  ┌────────────────────┐  ┌────────────────────────────┐    │
    │  │ BufferPoolManager  │  │  StorageManager + WAL      │    │
    │  │ (OpenClaw Cache)   │  │  (OpenClaw ResourceMgr)    │    │
    │  └────────────────────┘  └────────────────────────────┘    │
    │  ┌────────────────────┐  ┌────────────────────────────┐    │
    │  │ MemoryManager      │  │  EventBus (OpenClaw EvtMgr)│    │
    │  │ (SlabPool, Arena)  │  │  ProcessManager (schedlr)  │    │
    │  └────────────────────┘  └────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    data_dir           : Directory where database files are stored.
    page_pool_capacity : Number of 8-KiB page frames in the buffer pool.
    llm_api_key        : Optional API key to enable LLM-powered NL queries.
    llm_model          : LLM model name (default: gpt-4o-mini).
    llm_base_url       : LLM API base URL.
    """

    def __init__(
        self,
        data_dir: str | Path = "./llmdb_data",
        page_pool_capacity: int = 256,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Core subsystems (bottom-up initialisation)
        self._memory = MemoryManager(
            page_pool_capacity=page_pool_capacity,
            row_pool_capacity=page_pool_capacity * 16,
        )
        self._storage = StorageManager(self._data_dir)
        self._buffer = BufferPoolManager(
            self._memory, self._storage, num_frames=page_pool_capacity
        )
        self._catalog = Catalog(self._data_dir)
        self._events = reset_event_bus()
        self._processes = ProcessManager()

        # Query pipeline
        self._planner = Planner(self._catalog)
        self._executor = Executor(self._catalog)

        # Natural language
        _api_key = llm_api_key or os.environ.get("LLMDB_API_KEY")
        self._translator: NaturalLanguageTranslator = make_translator(
            api_key=_api_key,
            model=llm_model,
            base_url=llm_base_url,
        )

        # Start background process scheduler
        self._processes.start()

        # Crash recovery
        recovered = self._storage.recover()
        if recovered:
            self._log(f"WAL recovery: replayed {recovered} records")

    # ------------------------------------------------------------------
    # SQL query execution
    # ------------------------------------------------------------------

    def execute(self, sql: str) -> QueryResult:
        """Parse, plan, and execute a SQL statement."""
        self._events.post(QueryReceivedEvent(
            query_text=sql, session_id="default", is_natural_language=False
        ))
        t0 = time.monotonic()
        try:
            ast = parse_sql(sql)
            plan = self._planner.plan(ast)
            result = self._executor.execute(plan)
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            result = type("R", (), {
                "rows": [], "row_count": 0, "affected_rows": 0,
                "elapsed_ms": elapsed, "error": str(exc),
            })()

        self._events.queue(QueryResultEvent(
            session_id="default",
            rows=result.rows if hasattr(result, "rows") else [],
            row_count=result.row_count if hasattr(result, "row_count") else 0,
            affected_rows=result.affected_rows if hasattr(result, "affected_rows") else 0,
            elapsed_ms=result.elapsed_ms if hasattr(result, "elapsed_ms") else 0,
            error=result.error if hasattr(result, "error") else None,
        ))
        self._events.process_queued()

        if isinstance(result, ExecutionResult):
            return QueryResult(result, sql=sql)
        # Wrap duck-typed result
        from llmdb.query.executor import ExecutionResult as ER
        er = ER(
            rows=result.rows,
            row_count=result.row_count,
            affected_rows=result.affected_rows,
            elapsed_ms=result.elapsed_ms,
            error=result.error,
        )
        return QueryResult(er, sql=sql)

    # ------------------------------------------------------------------
    # Natural language interface
    # ------------------------------------------------------------------

    def ask(self, question: str) -> QueryResult:
        """Translate an English question into SQL and execute it.

        Examples
        --------
        >>> db.ask("How many users are there?")
        >>> db.ask("Show me all orders placed after January 2024")
        >>> db.ask("Who are the top 5 customers by total spend?")
        >>> db.ask("Delete all inactive accounts")
        """
        self._events.post(QueryReceivedEvent(
            query_text=question, session_id="default", is_natural_language=True
        ))
        schema_ctx = build_schema_context(self._catalog)
        translation = self._translator.translate(question, schema_ctx)
        result = self.execute(translation.sql)
        result._sql = translation.sql  # attach the generated SQL
        result._translation = translation  # attach full translation result
        return result

    def explain(self, question: str) -> str:
        """Translate a question to SQL and explain the reasoning without executing."""
        schema_ctx = build_schema_context(self._catalog)
        translation = self._translator.translate(question, schema_ctx)
        return str(translation)

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def create_table(self, name: str, columns: List[Dict[str, Any]]) -> None:
        """Create a table programmatically.

        Example
        -------
        >>> db.create_table("users", [
        ...     {"name": "id",    "type": "INTEGER", "primary_key": True},
        ...     {"name": "email", "type": "TEXT",    "nullable": False},
        ... ])
        """
        schema = {"name": name, "type": "table", "columns": columns}
        self._catalog.create_table(schema)
        self._events.post(SchemaChangedEvent("CREATE", "table", name))

    def tables(self) -> List[str]:
        """Return a list of all table names."""
        return self._catalog.list_tables()

    def describe(self, table_name: str) -> List[Dict[str, Any]]:
        """Describe the columns of a table."""
        obj = self._catalog.get_table(table_name)
        if obj is None:
            raise KeyError(f"Table '{table_name}' not found")
        from llmdb.core.actors import ColumnSetComponent
        col_comp = obj.get_component_typed(ColumnSetComponent)
        if col_comp is None:
            return []
        return [
            {
                "name": c.name,
                "type": c.col_type.name,
                "nullable": c.nullable,
                "primary_key": c.primary_key,
                "default": c.default,
            }
            for c in col_comp.columns
        ]

    # ------------------------------------------------------------------
    # System status — memory / buffer / storage report
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of all subsystems.

        The 'remaining_memory' key answers: "how much memory is still
        available for new page frames?"

        Returns a nested dict covering:
          - memory       : slab pool stats (page frames, row buffers, arenas)
          - buffer_pool  : frame usage, hit ratio, dirty frames
          - storage      : number of open relations, WAL LSN
          - catalog      : list of tables and row counts
          - processes    : number of active background processes
        """
        mem_report = self._memory.report()
        buf_report = self._buffer.stats()
        # "remaining memory" = free frames in the buffer pool × page size.
        # All slab frames are pre-allocated at pool construction time so
        # MemoryManager.remaining_page_memory() is always 0 once the buffer
        # pool is built.  The meaningful metric is how many of those frames
        # are not yet holding a cached page.
        remaining_bytes = buf_report["remaining_bytes"]
        return {
            "memory": {
                **mem_report,
                "remaining_page_memory_bytes": remaining_bytes,
                "remaining_page_memory_mb": round(remaining_bytes / (1024 * 1024), 2),
            },
            "buffer_pool": buf_report,
            "storage": {
                "data_dir": str(self._data_dir),
                "relations": self._storage.list_relations(),
                "wal_lsn": self._storage.wal.current_lsn,
            },
            "catalog": {
                "tables": {
                    t: self._catalog.estimated_row_count(t)
                    for t in self._catalog.list_tables()
                }
            },
            "processes": {
                "active_background_processes": self._processes.active_count
            },
        }

    def remaining_memory(self) -> int:
        """Return remaining buffer-pool memory in bytes (free frames × page size)."""
        return self._buffer.remaining_memory

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> int:
        """Flush all dirty buffer pages to disk."""
        return self._buffer.flush_all()

    def close(self) -> None:
        """Flush, stop background processes, and close all file handles."""
        self._buffer.flush_all()
        self._processes.stop()
        self._storage.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Database(dir={self._data_dir!r}, "
            f"tables={self.tables()}, "
            f"remaining_memory={self.remaining_memory() // 1024}KiB)"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        import logging
        logging.getLogger("llmdb").info(msg)
