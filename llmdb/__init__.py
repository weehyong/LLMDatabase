"""
LLMDatabase — A database management system reimagined with OpenClaw-inspired
actor/component/event architecture, tuned for agentic LLM applications.

Key subsystems:
  core.memory   — Memory pool allocator (cf. OpenClaw MemoryPool)
  core.storage  — Page-based storage manager + Write-Ahead Log
  core.buffer   — Clock-sweep buffer pool (cf. OpenClaw ResourceManager)
  core.actors   — Actor/Component system for Tables, Indexes, Constraints
  core.events   — Event bus for query pipeline stages
  core.process  — Cooperative process scheduler for async query execution
  query.sql     — SQL parser & executor
  query.natural — English→SQL translator (LLM-powered)
  query.planner — Cost-based query planner
  llm           — LLM integration helpers
  api           — Public-facing database client
"""

from llmdb.api.database import Database

__all__ = ["Database"]
__version__ = "0.1.0"
