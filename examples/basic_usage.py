"""
Basic Usage Example
===================
Shows the core LLMDatabase API: collections, documents, queries, and vector search.
"""

from llmdb import Database

# ── Open a database (in-memory for this example) ─────────────────────────────
db = Database(":memory:")

# ── Create a collection ───────────────────────────────────────────────────────
messages = db.create_collection("messages", vector_dim=3)

# ── Insert documents ──────────────────────────────────────────────────────────
doc1 = messages.insert(
    {"role": "user", "content": "Book a flight to Paris", "tokens": 8},
    tags=["conversation", "travel"],
    embedding=[0.9, 0.1, 0.0],
)
doc2 = messages.insert(
    {"role": "assistant", "content": "Sure! When would you like to fly?", "tokens": 9},
    tags=["conversation"],
    embedding=[0.8, 0.2, 0.0],
)
doc3 = messages.insert(
    {"role": "user", "content": "What is the weather today?", "tokens": 6},
    tags=["conversation", "weather"],
    embedding=[0.1, 0.1, 0.8],
)

print("=== All documents ===")
for d in messages.list():
    print(f"  [{d.data['role']}] {d.data['content'][:40]}")

# ── Structured query ──────────────────────────────────────────────────────────
print("\n=== User messages with < 8 tokens ===")
short_user = db.query(
    "messages",
    {"$and": [{"role": "user"}, {"tokens": {"$lt": 8}}]},
)
for d in short_user:
    print(f"  {d.data['content']}")

# ── Vector / semantic search ──────────────────────────────────────────────────
print("\n=== Semantic search: 'travel intent' ===")
results = db.vector_search("messages", [0.85, 0.15, 0.0], top_k=2)
for r in results:
    print(f"  score={r['score']:.3f}  {r['document']['data']['content'][:50]}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
db.close()
print("\nDone.")
