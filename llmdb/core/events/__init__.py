"""
Event Bus — inspired by OpenClaw's EventManager.

OpenClaw routed all inter-system communication through a typed event bus so
that game systems (physics, rendering, AI) remained decoupled.  Listeners
registered for specific event types; emitters simply posted events and never
knew who consumed them.

Here the same bus drives the query execution pipeline:

  QueryReceivedEvent   → parser consumes it → emits QueryParsedEvent
  QueryParsedEvent     → planner consumes it → emits QueryPlannedEvent
  QueryPlannedEvent    → executor consumes it → emits QueryResultEvent
  QueryResultEvent     → client / metrics handler

This keeps every stage testable in isolation and makes it trivial to add
cross-cutting concerns (tracing, caching, rate-limiting) as new listeners.
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar


# ---------------------------------------------------------------------------
# Base Event
# ---------------------------------------------------------------------------

class DbEvent(ABC):
    """Base class for all database events (cf. OpenClaw's IEventData)."""

    def __init__(self) -> None:
        self.event_id: str = str(uuid.uuid4())[:8]
        self.timestamp: float = time.monotonic()

    @property
    @abstractmethod
    def event_type(self) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.event_id})"


# ---------------------------------------------------------------------------
# Concrete Events — Query Pipeline
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class QueryReceivedEvent(DbEvent):
    """Fired when the user submits a query string (SQL or English)."""
    query_text: str
    session_id: str
    is_natural_language: bool = False

    def __post_init__(self) -> None:
        super().__init__()

    event_type = "query.received"


@dataclass(eq=False)
class QueryParsedEvent(DbEvent):
    """Fired after the parser produces an AST."""
    original_query: str
    ast: Any           # opaque AST node from the SQL parser
    session_id: str

    def __post_init__(self) -> None:
        super().__init__()

    event_type = "query.parsed"


@dataclass(eq=False)
class QueryPlannedEvent(DbEvent):
    """Fired after the query planner builds an execution plan."""
    session_id: str
    plan: Any          # opaque plan node from the planner
    estimated_cost: float = 0.0

    def __post_init__(self) -> None:
        super().__init__()

    event_type = "query.planned"


@dataclass(eq=False)
class QueryResultEvent(DbEvent):
    """Fired when a query finishes execution."""
    session_id: str
    rows: List[Dict[str, Any]]
    row_count: int
    affected_rows: int
    elapsed_ms: float
    error: Optional[str] = None

    def __post_init__(self) -> None:
        super().__init__()

    event_type = "query.result"


@dataclass(eq=False)
class SchemaChangedEvent(DbEvent):
    """Fired on DDL operations (CREATE TABLE, DROP TABLE, ALTER TABLE…)."""
    operation: str   # CREATE | DROP | ALTER
    object_type: str # table | index | view
    object_name: str

    def __post_init__(self) -> None:
        super().__init__()

    event_type = "schema.changed"


@dataclass(eq=False)
class BufferEvictionEvent(DbEvent):
    """Fired by the buffer pool when a dirty page is evicted."""
    relation: str
    page_no: int
    lsn: int

    def __post_init__(self) -> None:
        super().__init__()

    event_type = "buffer.eviction"


@dataclass(eq=False)
class TransactionEvent(DbEvent):
    """Fired on transaction boundaries (BEGIN, COMMIT, ROLLBACK)."""
    session_id: str
    operation: str  # BEGIN | COMMIT | ROLLBACK
    txn_id: str

    def __post_init__(self) -> None:
        super().__init__()

    event_type = "txn.boundary"


# ---------------------------------------------------------------------------
# Listener type alias
# ---------------------------------------------------------------------------

EventListener = Callable[[DbEvent], None]
E = TypeVar("E", bound=DbEvent)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """Thread-safe publish/subscribe event bus.

    API mirrors OpenClaw's IEventManager:
      subscribe(event_type, listener)
      unsubscribe(event_type, listener)
      post(event)          — immediate synchronous dispatch
      queue(event)         — defer to next process_queued() call
      process_queued()     — drain the deferred queue (called once per 'tick')

    The 'tick' concept is taken directly from OpenClaw's game loop where
    EventManager::VTick() was called each frame to drain events.  In our
    database engine, process_queued() is called at the end of each query
    execution to fire monitoring/logging events without blocking the query.
    """

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventListener]] = defaultdict(list)
        self._queue: List[DbEvent] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def subscribe(self, event_type: str, listener: EventListener) -> None:
        with self._lock:
            if listener not in self._listeners[event_type]:
                self._listeners[event_type].append(listener)

    def unsubscribe(self, event_type: str, listener: EventListener) -> None:
        with self._lock:
            try:
                self._listeners[event_type].remove(listener)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    def post(self, event: DbEvent) -> None:
        """Synchronously dispatch to all registered listeners."""
        with self._lock:
            listeners = list(self._listeners.get(event.event_type, []))
            listeners += list(self._listeners.get("*", []))  # wildcard
        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:
                # Events must not propagate exceptions back to the emitter
                import logging
                logging.getLogger("llmdb.events").warning(
                    "Listener %s raised: %s", listener, exc
                )

    def queue(self, event: DbEvent) -> None:
        """Enqueue an event for deferred dispatch."""
        with self._lock:
            self._queue.append(event)

    def process_queued(self, max_events: int = 100) -> int:
        """Drain and dispatch up to max_events from the deferred queue."""
        with self._lock:
            batch = self._queue[:max_events]
            self._queue = self._queue[max_events:]
        for event in batch:
            self.post(event)
        return len(batch)

    def listener_count(self, event_type: str = "*") -> int:
        with self._lock:
            if event_type == "*":
                return sum(len(v) for v in self._listeners.values())
            return len(self._listeners.get(event_type, []))


# ---------------------------------------------------------------------------
# Global bus singleton (tests / API code can replace this)
# ---------------------------------------------------------------------------

_global_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    global _global_bus
    with _bus_lock:
        if _global_bus is None:
            _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> EventBus:
    global _global_bus
    with _bus_lock:
        _global_bus = EventBus()
    return _global_bus
