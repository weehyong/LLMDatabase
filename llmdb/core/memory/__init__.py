"""
Memory Manager — inspired by OpenClaw's MemoryPool and ChunkedAllocator.

OpenClaw maintained a slab-based memory pool so that game objects (actors)
could be allocated and freed in O(1) without heap fragmentation.  Here the
same idea is applied to database pages and row buffers that live in the
buffer pool.

Components
----------
MemoryBlock  — a raw, fixed-size memory slab.
SlabPool     — a per-object-size free-list allocator.
MemoryArena  — a region-based allocator for short-lived query scratch space.
MemoryManager — singleton that owns all pools and reports remaining capacity.
"""

from __future__ import annotations

import ctypes
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# MemoryBlock
# ---------------------------------------------------------------------------

class MemoryBlock:
    """A fixed-size, reusable memory slab (analogous to OpenClaw's memory chunk).

    In OpenClaw a 'chunk' was a raw byte buffer acquired from a pool and
    returned when the game-object was destroyed.  Here the same slab is used
    to back a database page frame in the buffer pool.
    """

    def __init__(self, size: int, block_id: int) -> None:
        self._size = size
        self._block_id = block_id
        self._data = bytearray(size)
        self._in_use = False
        self._owner: Optional[str] = None  # e.g. "page:42"

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self._size

    @property
    def block_id(self) -> int:
        return self._block_id

    @property
    def in_use(self) -> bool:
        return self._in_use

    @property
    def data(self) -> bytearray:
        return self._data

    def acquire(self, owner: str) -> bytearray:
        if self._in_use:
            raise RuntimeError(f"Block {self._block_id} already in use by '{self._owner}'")
        self._in_use = True
        self._owner = owner
        self._data[:] = b"\x00" * self._size  # zero-fill on acquisition
        return self._data

    def release(self) -> None:
        self._in_use = False
        self._owner = None

    def __repr__(self) -> str:
        return (
            f"MemoryBlock(id={self._block_id}, size={self._size}, "
            f"in_use={self._in_use}, owner={self._owner!r})"
        )


# ---------------------------------------------------------------------------
# SlabPool
# ---------------------------------------------------------------------------

class SlabPool:
    """A per-slab-size free-list allocator.

    Inspired by the OpenClaw pattern of pre-allocating a fixed number of
    same-sized objects to avoid repeated malloc/free overhead.

    Parameters
    ----------
    slab_size:    size of each allocation unit in bytes
    capacity:     maximum number of slabs (hard limit — like VRAM budget)
    """

    def __init__(self, slab_size: int, capacity: int) -> None:
        self._slab_size = slab_size
        self._capacity = capacity
        self._lock = threading.Lock()
        self._all_blocks: List[MemoryBlock] = [
            MemoryBlock(slab_size, i) for i in range(capacity)
        ]
        self._free_list: List[MemoryBlock] = list(self._all_blocks)

    # ------------------------------------------------------------------
    @property
    def slab_size(self) -> int:
        return self._slab_size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._free_list)

    @property
    def used_count(self) -> int:
        return self._capacity - self.free_count

    @property
    def remaining_bytes(self) -> int:
        return self.free_count * self._slab_size

    # ------------------------------------------------------------------
    def allocate(self, owner: str) -> MemoryBlock:
        """Return a free slab or raise MemoryError if the pool is exhausted."""
        with self._lock:
            if not self._free_list:
                raise MemoryError(
                    f"SlabPool(size={self._slab_size}) exhausted — "
                    f"all {self._capacity} slabs are in use"
                )
            block = self._free_list.pop()
        block.acquire(owner)
        return block

    def free(self, block: MemoryBlock) -> None:
        """Return a slab to the free list."""
        block.release()
        with self._lock:
            self._free_list.append(block)

    def stats(self) -> Dict[str, int]:
        return {
            "slab_size": self._slab_size,
            "capacity": self._capacity,
            "used": self.used_count,
            "free": self.free_count,
            "remaining_bytes": self.remaining_bytes,
        }


# ---------------------------------------------------------------------------
# MemoryArena — scratch allocator for query execution
# ---------------------------------------------------------------------------

@dataclass
class MemoryArena:
    """Region-based allocator for short-lived query scratch space.

    All allocations inside a single query share one contiguous arena which
    is discarded in O(1) when the query finishes — no individual frees needed.
    This mirrors how OpenClaw used transient 'level scratch' memory.
    """

    capacity: int
    _buffer: bytearray = field(init=False)
    _offset: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._buffer = bytearray(self.capacity)

    def allocate(self, size: int) -> memoryview:
        if self._offset + size > self.capacity:
            raise MemoryError(
                f"MemoryArena exhausted: requested {size} bytes but only "
                f"{self.capacity - self._offset} remain"
            )
        view = memoryview(self._buffer)[self._offset : self._offset + size]
        self._offset += size
        return view

    def reset(self) -> None:
        """Reclaim all arena memory in O(1)."""
        self._offset = 0

    @property
    def used_bytes(self) -> int:
        return self._offset

    @property
    def remaining_bytes(self) -> int:
        return self.capacity - self._offset


# ---------------------------------------------------------------------------
# MemoryManager — singleton registry (cf. OpenClaw GameLogic memory budget)
# ---------------------------------------------------------------------------

class MemoryManager:
    """Central memory authority for the database engine.

    OpenClaw's GameLogic class tracked all live game objects and enforced a
    hard memory budget.  MemoryManager does the same for page-frame slabs,
    index-node slabs, and row-buffer slabs.

    Usage
    -----
    >>> mm = MemoryManager(page_size=8192, page_pool_capacity=1024)
    >>> mm.remaining_page_memory()     # bytes still available for page frames
    >>> block = mm.alloc_page_frame("page:42")
    >>> mm.free_page_frame(block)
    """

    def __init__(
        self,
        page_size: int = 8192,
        page_pool_capacity: int = 256,
        row_slab_size: int = 256,
        row_pool_capacity: int = 4096,
        arena_size: int = 4 * 1024 * 1024,  # 4 MiB per query
    ) -> None:
        self._page_pool = SlabPool(page_size, page_pool_capacity)
        self._row_pool = SlabPool(row_slab_size, row_pool_capacity)
        self._arena_size = arena_size
        self._arenas: List[MemoryArena] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Page-frame allocation (used by the buffer pool manager)
    # ------------------------------------------------------------------

    def alloc_page_frame(self, owner: str) -> MemoryBlock:
        return self._page_pool.allocate(owner)

    def free_page_frame(self, block: MemoryBlock) -> None:
        self._page_pool.free(block)

    def remaining_page_memory(self) -> int:
        """Remaining bytes available for page frames."""
        return self._page_pool.remaining_bytes

    # ------------------------------------------------------------------
    # Row-buffer allocation (used by query executors)
    # ------------------------------------------------------------------

    def alloc_row_buffer(self, owner: str) -> MemoryBlock:
        return self._row_pool.allocate(owner)

    def free_row_buffer(self, block: MemoryBlock) -> None:
        self._row_pool.free(block)

    # ------------------------------------------------------------------
    # Query scratch arenas
    # ------------------------------------------------------------------

    def acquire_arena(self) -> MemoryArena:
        with self._lock:
            if self._arenas:
                arena = self._arenas.pop()
                arena.reset()
                return arena
        return MemoryArena(capacity=self._arena_size)

    def release_arena(self, arena: MemoryArena) -> None:
        arena.reset()
        with self._lock:
            self._arenas.append(arena)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def report(self) -> Dict[str, object]:
        return {
            "page_pool": self._page_pool.stats(),
            "row_pool": self._row_pool.stats(),
            "cached_arenas": len(self._arenas),
        }
