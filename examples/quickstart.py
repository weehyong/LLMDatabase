"""
Quick-start example: building a task tracker with LLMDatabase.

Demonstrates:
  1. Schema creation
  2. Data ingestion
  3. SQL queries
  4. Natural language queries (English → SQL)
  5. Memory / system status report
"""

from llmdb import Database

# ------------------------------------------------------------------
# 1. Open / create database
# ------------------------------------------------------------------
db = Database("./example_db")

# ------------------------------------------------------------------
# 2. Create tables (DDL)
# ------------------------------------------------------------------
db.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id   INTEGER PRIMARY KEY,
        name TEXT    NOT NULL,
        status TEXT  DEFAULT 'active'
    )
""")

db.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id         INTEGER PRIMARY KEY,
        project_id INTEGER NOT NULL,
        title      TEXT    NOT NULL,
        priority   INTEGER DEFAULT 3,
        done       INTEGER DEFAULT 0
    )
""")

# ------------------------------------------------------------------
# 3. Insert data (DML)
# ------------------------------------------------------------------
db.execute("INSERT INTO projects (id, name) VALUES (1, 'LLMDatabase')")
db.execute("INSERT INTO projects (id, name) VALUES (2, 'Agent Framework')")

tasks = [
    (1, 1, "Memory manager", 1, 1),
    (2, 1, "Storage manager", 1, 1),
    (3, 1, "Buffer pool", 2, 0),
    (4, 1, "SQL parser", 2, 1),
    (5, 1, "NL translator", 1, 0),
    (6, 2, "Agent loop", 2, 0),
    (7, 2, "Tool registry", 3, 0),
]
for t in tasks:
    db.execute(f"INSERT INTO tasks (id, project_id, title, priority, done) VALUES {t}")

# ------------------------------------------------------------------
# 4. SQL Queries
# ------------------------------------------------------------------
print("=== All projects ===")
for row in db.execute("SELECT * FROM projects").rows:
    print(" ", row)

print("\n=== High-priority tasks ===")
for row in db.execute("SELECT * FROM tasks WHERE priority <= 2 ORDER BY priority").rows:
    print(" ", row)

print("\n=== Incomplete tasks ===")
result = db.execute("SELECT * FROM tasks WHERE done = 0")
for row in result.rows:
    print(" ", row)
print(f"  ({result.row_count} incomplete tasks)")

# ------------------------------------------------------------------
# 5. Natural Language Queries
# ------------------------------------------------------------------
print("\n=== English → SQL ===")

questions = [
    "How many tasks are there?",
    "Show me all tasks that are not done",
    "What is the total priority of tasks?",
    "Top 3 tasks by priority",
]

for q in questions:
    print(f"\n  Q: {q!r}")
    result = db.ask(q)
    print(f"  SQL: {result.sql}")
    if result.error:
        print(f"  Error: {result.error}")
    else:
        print(f"  Result: {result.rows} ({result.row_count} rows)")

# ------------------------------------------------------------------
# 6. System status
# ------------------------------------------------------------------
print("\n=== System Status ===")
status = db.status()
print(f"  Remaining page memory : {status['memory']['remaining_page_memory_mb']} MB")
print(f"  Buffer pool hit ratio : {status['buffer_pool']['hit_ratio']:.0%}")
print(f"  WAL LSN               : {status['storage']['wal_lsn']}")
print(f"  Tables                : {list(status['catalog']['tables'].keys())}")

db.close()
print("\nDone.")
