"""
Cooperative Process Scheduler — inspired by OpenClaw's ProcessManager.

OpenClaw's ProcessManager ran a list of *Process* objects each game tick.
Processes could be chained (one starts when another finishes), paused,
resumed, or killed.  This was used for animated sequences, timed events,
and multi-step game logic without threads.

In the database engine the same idea handles:
  • Multi-phase query execution (parse → plan → execute → fetch)
  • Background tasks (vacuum/garbage-collect, checkpoint, statistics)
  • Async result streaming to the client

Differences from OpenClaw
--------------------------
  - We use actual threading instead of a game-loop tick so that background
    processes genuinely run concurrently (database needs real parallelism).
  - The chaining and lifecycle API remains faithful to the original.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

log = logging.getLogger("llmdb.process")


# ---------------------------------------------------------------------------
# ProcessState
# ---------------------------------------------------------------------------

class ProcessState(Enum):
    UNINITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    ABORTED = auto()


# ---------------------------------------------------------------------------
# Process — base class (cf. OpenClaw's Process)
# ---------------------------------------------------------------------------

class Process(ABC):
    """A cooperative unit of work that can be chained and managed.

    Sub-classes override on_init(), on_update(), on_success(), on_fail(),
    on_abort() — exactly as in OpenClaw's Process hierarchy.
    """

    def __init__(self) -> None:
        self._id = str(uuid.uuid4())[:8]
        self._state = ProcessState.UNINITIALIZED
        self._child: Optional["Process"] = None

    # ------------------------------------------------------------------
    @property
    def process_id(self) -> str:
        return self._id

    @property
    def state(self) -> ProcessState:
        return self._state

    @property
    def is_alive(self) -> bool:
        return self._state in (ProcessState.RUNNING, ProcessState.PAUSED)

    @property
    def is_dead(self) -> bool:
        return self._state in (
            ProcessState.SUCCEEDED, ProcessState.FAILED, ProcessState.ABORTED
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks (override in sub-classes)
    # ------------------------------------------------------------------

    def on_init(self) -> None:
        """Called once when the process first runs."""

    @abstractmethod
    def on_update(self) -> None:
        """Called every tick while the process is RUNNING."""

    def on_success(self) -> None:
        """Called when the process transitions to SUCCEEDED."""

    def on_fail(self) -> None:
        """Called when the process transitions to FAILED."""

    def on_abort(self) -> None:
        """Called when the process is forcibly terminated."""

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def succeed(self) -> None:
        self._state = ProcessState.SUCCEEDED

    def fail(self) -> None:
        self._state = ProcessState.FAILED

    def pause(self) -> None:
        if self._state == ProcessState.RUNNING:
            self._state = ProcessState.PAUSED

    def resume(self) -> None:
        if self._state == ProcessState.PAUSED:
            self._state = ProcessState.RUNNING

    # ------------------------------------------------------------------
    # Chaining
    # ------------------------------------------------------------------

    def attach_child(self, child: "Process") -> "Process":
        """Chain another process to run when this one succeeds."""
        self._child = child
        return child

    @property
    def child(self) -> Optional["Process"]:
        return self._child

    def remove_child(self) -> Optional["Process"]:
        c = self._child
        self._child = None
        return c

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._id}, state={self._state.name})"


# ---------------------------------------------------------------------------
# Built-in process types
# ---------------------------------------------------------------------------

class DelayProcess(Process):
    """Waits for `seconds` before succeeding (used to schedule background work)."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self._duration = seconds
        self._deadline: float = 0.0

    def on_init(self) -> None:
        self._deadline = time.monotonic() + self._duration

    def on_update(self) -> None:
        if time.monotonic() >= self._deadline:
            self.succeed()


class CallbackProcess(Process):
    """Wraps a callable as a one-shot process."""

    def __init__(self, fn: Callable[[], None]) -> None:
        super().__init__()
        self._fn = fn

    def on_update(self) -> None:
        try:
            self._fn()
            self.succeed()
        except Exception as exc:
            log.exception("CallbackProcess %s failed: %s", self._id, exc)
            self.fail()


class LoopProcess(Process):
    """Calls a function every `interval` seconds until manually aborted."""

    def __init__(self, fn: Callable[[], None], interval: float) -> None:
        super().__init__()
        self._fn = fn
        self._interval = interval
        self._next_run: float = 0.0

    def on_init(self) -> None:
        self._next_run = time.monotonic()

    def on_update(self) -> None:
        now = time.monotonic()
        if now >= self._next_run:
            try:
                self._fn()
            except Exception as exc:
                log.exception("LoopProcess %s error: %s", self._id, exc)
            self._next_run = now + self._interval


# ---------------------------------------------------------------------------
# ProcessManager
# ---------------------------------------------------------------------------

class ProcessManager:
    """Owns and advances all active processes.

    The manager runs in a background daemon thread, calling on_update() on
    each live process in order.  When a process succeeds its child (if any)
    is automatically attached and started — exactly as in OpenClaw.

    Usage
    -----
    >>> pm = ProcessManager()
    >>> pm.start()
    >>> proc = pm.attach(CallbackProcess(lambda: print("done")))
    >>> pm.stop()
    """

    def __init__(self, tick_interval: float = 0.05) -> None:
        self._tick_interval = tick_interval
        self._processes: List[Process] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    def attach(self, process: Process) -> Process:
        """Add a process to the manager.  Returns the process for chaining."""
        with self._lock:
            self._processes.append(process)
        return process

    def abort_all(self) -> None:
        with self._lock:
            procs = list(self._processes)
        for p in procs:
            if p.is_alive:
                p.on_abort()
                p._state = ProcessState.ABORTED  # noqa: SLF001

    def abort(self, process_id: str) -> bool:
        with self._lock:
            for p in self._processes:
                if p.process_id == process_id and p.is_alive:
                    p.on_abort()
                    p._state = ProcessState.ABORTED  # noqa: SLF001
                    return True
        return False

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ProcessManager")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while self._running:
            self._tick()
            time.sleep(self._tick_interval)

    def _tick(self) -> None:
        """Advance all processes by one tick (cf. OpenClaw's ProcessManager::UpdateProcesses)."""
        dead: List[Process] = []
        new_children: List[Process] = []

        with self._lock:
            procs = list(self._processes)

        for proc in procs:
            if proc.state == ProcessState.UNINITIALIZED:
                proc.on_init()
                proc._state = ProcessState.RUNNING  # noqa: SLF001

            if proc.state == ProcessState.RUNNING:
                try:
                    proc.on_update()
                except Exception as exc:
                    log.exception("Process %s raised: %s", proc.process_id, exc)
                    proc._state = ProcessState.FAILED  # noqa: SLF001

            if proc.is_dead:
                if proc.state == ProcessState.SUCCEEDED:
                    proc.on_success()
                    child = proc.remove_child()
                    if child:
                        new_children.append(child)
                elif proc.state == ProcessState.FAILED:
                    proc.on_fail()
                dead.append(proc)

        with self._lock:
            for d in dead:
                try:
                    self._processes.remove(d)
                except ValueError:
                    pass
            self._processes.extend(new_children)

    # ------------------------------------------------------------------
    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._processes)

    def __repr__(self) -> str:
        return f"ProcessManager(active={self.active_count}, running={self._running})"
