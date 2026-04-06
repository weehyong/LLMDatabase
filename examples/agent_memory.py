"""
Agent Memory Example
====================
Demonstrates how an AI agent would use LLMDatabase to manage its memory:
episodic (events), semantic (facts), and procedural (skills/how-to).
"""

from llmdb import Database, MemoryType

db = Database(":memory:")

# ── Episodic memory: things that happened ─────────────────────────────────────
db.memory.add(
    "The user asked me to book a flight to Paris on 2025-03-12.",
    memory_type=MemoryType.EPISODIC,
    importance=0.7,
    tags=["user:alice", "task:travel"],
    embedding=[0.9, 0.05, 0.05],
)
db.memory.add(
    "I failed to retrieve the user's passport number — it is not in the DB.",
    memory_type=MemoryType.EPISODIC,
    importance=0.4,
    tags=["user:alice", "error"],
    embedding=[0.1, 0.8, 0.1],
)

# ── Semantic memory: world knowledge ─────────────────────────────────────────
db.memory.add(
    "Paris is the capital of France.",
    memory_type=MemoryType.SEMANTIC,
    importance=0.3,
    tags=["geography"],
    embedding=[0.2, 0.2, 0.6],
)
db.memory.add(
    "The user's preferred airline is Air France.",
    memory_type=MemoryType.SEMANTIC,
    importance=0.8,
    tags=["user:alice", "preferences"],
    embedding=[0.85, 0.1, 0.05],
)

# ── Procedural memory: skills / how-to ───────────────────────────────────────
db.memory.add(
    "To book a flight, call POST /api/flights with {origin, destination, date, passenger_id}.",
    memory_type=MemoryType.PROCEDURAL,
    importance=0.9,
    tags=["api", "travel"],
    embedding=[0.6, 0.3, 0.1],
)

# ── List all memories by importance ──────────────────────────────────────────
print("=== All memories (by importance) ===")
for mem in db.memory.list():
    print(f"  [{mem.memory_type.value:10s}] ({mem.importance:.1f}) {mem.content[:60]}")

# ── Filter by type ────────────────────────────────────────────────────────────
print("\n=== Procedural memories ===")
for mem in db.memory.list(memory_type=MemoryType.PROCEDURAL):
    print(f"  {mem.content}")

# ── Semantic search: find memories relevant to a travel booking task ──────────
print("\n=== Semantic search: 'help me book travel' ===")
query_vec = [0.88, 0.08, 0.04]  # travel-flavoured vector
results = db.memory.search(query_vec, top_k=3)
for mem, score in results:
    print(f"  score={score:.3f}  [{mem.memory_type.value}] {mem.content[:60]}")

# ── High-importance filter ───────────────────────────────────────────────────
print("\n=== Critical memories (importance >= 0.7) ===")
for mem in db.memory.list(min_importance=0.7):
    print(f"  ({mem.importance:.1f}) {mem.content[:60]}")

db.close()
print("\nDone.")
