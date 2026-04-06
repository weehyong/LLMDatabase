# LLMDatabase

> A database management system **reimagined from scratch for agentic applications**.

AI agents have fundamentally different data-access patterns than traditional applications.
They need to store and recall **memories**, perform **semantic search** over rich context,
handle **flexible schemas** that evolve at runtime, and do all of this through an
API that an LLM can reason about easily.

LLMDatabase is a lightweight, embeddable database that addresses every one of those
requirements out of the box.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Collections & Documents** | Schema-less JSON documents grouped into named collections |
| **Structured Queries** | MongoDB-style filter operators: `$eq`, `$gt`, `$in`, `$contains`, `$and`, `$or`, `$not` … |
| **Vector / Semantic Search** | Store embeddings alongside documents and search by cosine similarity |
| **Agent Memory** | Three first-class memory types — *episodic*, *semantic*, *procedural* |
| **Importance & Tags** | Every memory carries an importance score and a tag list for quick retrieval |
| **REST API** | FastAPI-based HTTP server with OpenAPI docs at `/docs` |
| **CLI** | Full-featured command-line interface (`llmdb`) |
| **Embeddable** | Use directly as a Python library (SQLite under the hood, zero extra services) |

---

## 📦 Installation

```bash
pip install -e ".[dev]"   # from source with dev dependencies
```

---

## 🚀 Quick Start

### Python API

```python
from llmdb import Database

db = Database("myagent.llmdb")

# Create a collection
messages = db.create_collection("messages", vector_dim=384)

# Insert a document with an embedding
doc = messages.insert(
    {"role": "user", "content": "Book a flight to Paris"},
    tags=["travel"],
    embedding=[...],   # your embedding vector here
)

# Structured query
results = db.query(
    "messages",
    {"$and": [{"role": "user"}, {"tokens": {"$lt": 50}}]},
    sort_by="created_at",
)

# Semantic search
hits = db.vector_search("messages", query_embedding=[...], top_k=5)

# Agent memory
db.memory.add(
    "User prefers Air France for long-haul flights.",
    memory_type="semantic",
    importance=0.85,
    tags=["preferences", "user:alice"],
)
relevant = db.memory.search(query_embedding=[...], top_k=3)

db.close()
```

### REST API

```bash
# Start the server
llmdb --db myagent.llmdb serve --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn "llmdb.api.server:create_app" --factory --port 8000
```

Interactive docs are available at **http://localhost:8000/docs**.

### CLI

```bash
# Collections
llmdb collection create posts
llmdb collection list
llmdb collection drop posts

# Documents
llmdb --db mydb.llmdb doc insert posts '{"title":"Hello","author":"alice"}'
llmdb --db mydb.llmdb doc list posts --limit 10
llmdb --db mydb.llmdb doc query posts '{"author":{"$eq":"alice"}}'

# Memory
llmdb --db mydb.llmdb memory add "User prefers dark mode" --type semantic --importance 0.8
llmdb --db mydb.llmdb memory list --type semantic
```

---

## 🧠 Memory Types

| Type | Purpose | Example |
|---|---|---|
| `episodic` | Specific events and experiences | "User asked to book a flight on 2025-03-12" |
| `semantic` | General facts and world knowledge | "Paris is the capital of France" |
| `procedural` | Skills and how-to knowledge | "To book a flight, call POST /api/flights …" |

---

## 🔍 Query Language

```python
# Equality (shorthand)
{"author": "alice"}

# Comparison operators
{"views": {"$gt": 100}}
{"score": {"$lte": 0.5}}
{"status": {"$ne": "draft"}}

# Set membership
{"tag": {"$in": ["ai", "ml", "llm"]}}

# String contains (case-insensitive)
{"content": {"$contains": "python"}}

# Field presence
{"embedding": {"$exists": True}}

# Logical combinators
{"$and": [{"author": "alice"}, {"views": {"$gt": 100}}]}
{"$or":  [{"status": "published"}, {"featured": True}]}
{"$not": {"draft": True}}

# Nested fields (dot notation)
{"meta.author.name": "alice"}
```

---

## 🗂 Project Structure

```
llmdb/
  __init__.py          Public API surface
  engine.py            Database façade (primary entry point)
  exceptions.py        Custom exceptions
  core/
    storage.py         SQLite storage engine
    collection.py      Collection + document CRUD
    document.py        Document model
  memory/
    store.py           MemoryStore (episodic / semantic / procedural)
  vector/
    index.py           Cosine-similarity vector index
  query/
    engine.py          Structured query engine
  api/
    server.py          FastAPI application factory
    models.py          Pydantic request/response models
    routes/
      collections.py   /collections endpoints
      documents.py     /collections/{name}/documents endpoints
      memory.py        /memory endpoints
  cli/
    main.py            Click CLI
tests/                 pytest test suite
examples/              Runnable usage examples
```

---

## 🧪 Running Tests

```bash
pip install -e ".[dev]"
pytest
```

---

## 📄 License

[MIT](LICENSE)
