# LLMDatabase

**LLMDatabase** reimagines a database management system from scratch for agentic LLM applications, drawing architectural inspiration from the open-source [OpenClaw](https://github.com/pjasicek/OpenClaw) game-engine codebase.

---

## Concept: OpenClaw → Database

OpenClaw (a C++ reimplementation of the 1997 game *Captain Claw*) pioneered a clean, layered architecture:

| OpenClaw Component | OpenClaw Purpose | LLMDatabase Equivalent |
|---|---|---|
| `ResourceManager` | Load, cache, evict game assets (sprites, audio) | **Buffer Pool Manager** — cache, evict database pages |
| `MemoryPool` / slab allocator | Pre-allocate fixed-size game-object buffers | **Memory Manager** — slab pools for page frames & row buffers |
| `Actor` + `Component` system | Composable game entities (player, enemy, item…) | **DbObject** + **DbComponent** — composable tables, indexes, constraints |
| `ActorFactory` (XML blueprints) | Create actors from declarative XML schemas | **SchemaFactory** (dict blueprints) — create tables from schema dicts |
| `EventManager` | Decouple game systems via typed events | **EventBus** — decouple query pipeline stages via typed events |
| `ProcessManager` | Cooperative scheduler for game logic processes | **ProcessManager** — background tasks (checkpoint, vacuum, stats) |
| Level / save-state files | Structured binary game-state persistence | **StorageManager** + **WAL** — page-based heap files + write-ahead log |

The result is a database where every concept maps cleanly to a well-understood game-engine abstraction — making the architecture easy to reason about and extend.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Database API  (llmdb.api)                                        │
│  ┌──────────────────┐  ┌────────────────────────────────────┐    │
│  │ NL Translator    │  │  SQL Parser → Planner → Executor   │    │
│  │ (English → SQL)  │  │  (llmdb.query.{sql,planner,exec})  │    │
│  └──────────────────┘  └────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Catalog  (system registry — cf. OpenClaw GameLogic)       │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────┐  ┌───────────────────────────────┐    │
│  │  Buffer Pool Manager  │  │  Storage Manager + WAL        │    │
│  │  Clock-Sweep eviction │  │  Heap files (.ldb) + wal.log  │    │
│  │  (cf. ResourceCache)  │  │  (cf. ResourceManager I/O)    │    │
│  └───────────────────────┘  └───────────────────────────────┘    │
│  ┌───────────────────────┐  ┌───────────────────────────────┐    │
│  │  Memory Manager       │  │  EventBus + ProcessManager    │    │
│  │  SlabPool / Arena     │  │  (cf. EventManager + Process) │    │
│  └───────────────────────┘  └───────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### Subsystems

#### 1. Memory Manager (`llmdb.core.memory`)
Pre-allocates fixed-size memory slabs so page frames and row buffers are handed out in O(1) with zero heap fragmentation — directly inspired by OpenClaw's `MemoryPool` / `ChunkedAllocator`.  
- **`SlabPool`** — per-size free-list allocator  
- **`MemoryArena`** — region allocator for per-query scratch space (reset in O(1))  
- **`MemoryManager`** — singleton that owns all pools and reports *remaining memory*

#### 2. Storage Manager (`llmdb.core.storage`)
The only component that performs real disk I/O, mirroring OpenClaw's `ResourceManager`.  
- **`HeapFile`** — a paged binary file (`*.ldb`), one per relation  
- **`WALManager`** — append-only Write-Ahead Log for crash recovery  
- **`Page`** — an 8 KiB page with a 16-byte header (magic, LSN, slot count)  
- **`StorageManager`** — routes all `read_page` / `write_page` calls through the WAL

#### 3. Buffer Pool Manager (`llmdb.core.buffer`)
An in-memory cache of page frames — exactly OpenClaw's `ResourceCache`, re-targeted at database pages.  
- **Clock-Sweep** eviction (as in PostgreSQL)  
- **Pinning** via `PinnedPage` RAII context manager  
- Reports *remaining buffer memory* (free frames × page size)

#### 4. Actor / Component System (`llmdb.core.actors`)
Tables, indexes, and constraints are **actors** with swappable **components** — OpenClaw's entity system applied to schema objects.  
- `ColumnSetComponent` — schema definition (analogous to `TransformComponent`)  
- `PrimaryKeyComponent` — PK constraint  
- `IndexComponent` — B-tree index (sorted in-memory, disk-backed in future)  
- `ForeignKeyComponent`, `CheckComponent`, `TriggerComponent`  
- **`SchemaFactory`** — creates actors from dict blueprints (cf. `ActorFactory` + XML)

#### 5. Event Bus (`llmdb.core.events`)
Typed publish/subscribe bus, mirroring OpenClaw's `EventManager`. Decouples query pipeline stages and enables cross-cutting concerns (logging, metrics) as listeners.  
- Synchronous `post()` and deferred `queue()` / `process_queued()` (cf. `VTick()`)  
- Wildcard `"*"` subscription for monitoring  
- Events: `QueryReceived`, `QueryParsed`, `QueryPlanned`, `QueryResult`, `SchemaChanged`, `BufferEviction`, `TransactionEvent`

#### 6. Process Manager (`llmdb.core.process`)
Cooperative process scheduler inspired by OpenClaw's `ProcessManager` and `Process` base class. Runs a background daemon thread, advancing processes each tick.  
- **Process chaining** — attach a child process that runs on parent success  
- **`DelayProcess`**, **`CallbackProcess`**, **`LoopProcess`** built-ins  
- Used for: checkpointing, statistics gathering, vacuum, async result streaming

#### 7. SQL Query Pipeline (`llmdb.query`)
- **Parser** (`llmdb.query.sql`) — recursive-descent SQL parser covering `SELECT`, `INSERT`, `UPDATE`, `DELETE`, DDL, transactions, JOINs, subexpressions, aggregates  
- **Planner** (`llmdb.query.planner`) — cost-based planner, chooses between SeqScan and IndexScan using estimated row counts  
- **Executor** (`llmdb.query.executor`) — Volcano iterator model; supports Filter, Projection, Sort, Limit, NestedLoopJoin, aggregate functions, CHECK constraints, BEFORE/AFTER triggers

#### 8. English → SQL Natural Language Interface (`llmdb.query.natural`)
**The evolution of SQL**: users write plain English instead of memorising syntax.

```python
db.ask("How many users are there?")
db.ask("Show me the top 10 customers by total spend")
db.ask("Delete all orders older than 30 days")
```

Two backends:  
- **`RuleBasedTranslator`** — zero-dependency keyword-matching fallback (works out of the box)  
- **`LLMTranslator`** — calls any OpenAI-compatible API; schema-aware, chain-of-thought reasoning, returns confidence + alternatives

---

## Quick Start

```bash
pip install -e .
```

```python
from llmdb import Database

db = Database("./mydb")

# SQL
db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
db.execute("INSERT INTO users (id, name, score) VALUES (1, 'Alice', 92.5)")
db.execute("INSERT INTO users (id, name, score) VALUES (2, 'Bob', 78.0)")

result = db.execute("SELECT * FROM users WHERE score > 80 ORDER BY score DESC")
print(result.rows)   # [{'id': 1, 'name': 'Alice', 'score': 92.5}]

# English (no API key needed for rule-based)
result = db.ask("How many users are there?")
print(result.rows)   # [{'count': 2}]
print(result.sql)    # SELECT COUNT(*) AS count FROM users

result = db.ask("Top 1 users by score")
print(result.rows)   # [{'id': 1, 'name': 'Alice', 'score': 92.5}]

# System status
status = db.status()
print(status["memory"]["remaining_page_memory_mb"], "MB free")
print(status["buffer_pool"]["hit_ratio"])

db.close()
```

### LLM-powered English queries (optional)

```bash
export LLMDB_API_KEY=sk-...
```

```python
db = Database("./mydb", llm_api_key="sk-...")
result = db.ask("Who are the top 5 customers by total order value since January 2024?")
print(result.sql)         # full SQL query
print(result._translation.explanation)  # reasoning
```

### Interactive REPL

```bash
python -m llmdb.cli --data-dir ./mydb
# or:
llmdb --data-dir ./mydb --api-key sk-...
```

```
llmdb> CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)
llmdb> INSERT INTO products (id, name, price) VALUES (1, 'Widget', 9.99)
llmdb> ?How many products cost less than 20?
  → SQL: SELECT COUNT(*) AS count FROM products WHERE price < 20
  → Confidence: 75%
  count
  -----
  1
llmdb> .status
llmdb> .describe products
llmdb> .quit
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 44 tests pass, covering memory pools, storage, buffer pool, actor/component, event bus, process manager, SQL parser, NL translator, and full integration.

---

## Project Structure

```
llmdb/
├── __init__.py          # Package entry point
├── cli.py               # Interactive REPL
├── api/                 # Public Database API
├── core/
│   ├── memory/          # Memory Manager (SlabPool, Arena)
│   ├── storage/         # Storage Manager (HeapFile, WAL)
│   ├── buffer/          # Buffer Pool Manager (Clock-Sweep)
│   ├── actors/          # Actor/Component system
│   ├── events/          # Event Bus
│   ├── process/         # Process Manager
│   └── catalog.py       # System Catalog
├── query/
│   ├── sql/             # SQL Parser (recursive-descent)
│   ├── planner/         # Query Planner
│   ├── natural/         # English → SQL Translator
│   └── executor.py      # Query Executor (Volcano model)
└── llm/                 # LLM client helpers
tests/
examples/
    quickstart.py
```

---

## Design Principles

1. **OpenClaw mapping** — every DBMS subsystem maps to a named game-engine concept so the architecture is self-documenting.
2. **Zero required dependencies** — the full SQL + rule-based NL stack works with Python stdlib only.
3. **Pluggable LLM** — drop in any OpenAI-compatible API for full natural-language power.
4. **Agentic-first** — the `Database.ask()` API is designed for LLM agents that need to query data in natural language without generating SQL manually.
5. **Correctness before optimisation** — the executor is correct and testable; performance can be layered on without changing interfaces.
