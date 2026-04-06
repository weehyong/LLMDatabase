"""
Buffer Pool Manager — inspired by OpenClaw's ResourceCache.

OpenClaw's ResourceCache kept a bounded LRU cache of decoded game resources
(sprites, audio, maps) and evicted the least-recently-used entry whenever
it needed to load a new one.  The buffer pool does the same for *database
pages*:

  • It allocates a fixed number of *frames* (page-sized memory slabs provided
    by the MemoryManager).
  • When a page is requested, the buffer pool checks its frame table first
    (cache hit) and only falls back to StorageManager.read_page on a miss.
  • Eviction uses the **Clock-Sweep** algorithm (as in PostgreSQL) — a
    rotating hand sweeps frames and clears reference bits; the first frame
    with a clear bit and no pinners is the victim.
  • Dirty frames are written back to storage before eviction (write-back
    cache), with the LSN recorded so the WAL stays consistent.

Key concepts mapped from OpenClaw
----------------------------------
ResourceCache  → BufferPoolManager
Resource       → BufferFrame   (a page-sized memory slab + metadata)
LRU eviction   → Clock-sweep eviction
Cache miss     → Disk I/O via StorageManager
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from llmdb.core.memory import MemoryBlock, MemoryManager
from llmdb.core.storage import Page, StorageManager


# ---------------------------------------------------------------------------
# PageId — globally unique page identifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=True)
class PageId:
    """Composite key: (relation_name, page_number)."""
    relation: str
    page_no: int

    def __str__(self) -> str:
        return f"{self.relation}#{self.page_no}"


# ---------------------------------------------------------------------------
# BufferFrame — one slot in the buffer pool
# ---------------------------------------------------------------------------

class BufferFrame:
    """A single buffer-pool frame (cf. OpenClaw's CachedResource).

    Holds a MemoryBlock (the raw bytes) together with bookkeeping metadata
    used by the eviction clock and the pin-count latch.
    """

    __slots__ = (
        "_frame_id", "_block", "_page_id", "_page",
        "_pin_count", "_dirty", "_ref_bit",
    )

    def __init__(self, frame_id: int, block: MemoryBlock) -> None:
        self._frame_id = frame_id
        self._block = block
        self._page_id: Optional[PageId] = None
        self._page: Optional[Page] = None
        self._pin_count: int = 0
        self._dirty: bool = False
        self._ref_bit: bool = False

    # ------------------------------------------------------------------
    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def page_id(self) -> Optional[PageId]:
        return self._page_id

    @property
    def page(self) -> Optional[Page]:
        return self._page

    @property
    def pin_count(self) -> int:
        return self._pin_count

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def ref_bit(self) -> bool:
        return self._ref_bit

    @property
    def block(self) -> MemoryBlock:
        return self._block

    # ------------------------------------------------------------------
    def load(self, page_id: PageId, raw: bytearray) -> Page:
        """Install a new page into this frame."""
        self._page_id = page_id
        # Copy raw bytes into the owned memory block
        self._block.data[:] = raw
        self._page = Page(page_id.page_no, self._block.data)
        self._dirty = False
        self._ref_bit = True
        self._pin_count = 1
        return self._page

    def pin(self) -> None:
        self._pin_count += 1
        self._ref_bit = True

    def unpin(self, dirty: bool = False) -> None:
        if self._pin_count == 0:
            raise RuntimeError(f"Frame {self._frame_id}: unpin on unpinned frame")
        self._pin_count -= 1
        if dirty:
            self._dirty = True

    def clear_ref(self) -> None:
        self._ref_bit = False

    def evict(self) -> None:
        """Reset frame to empty state."""
        self._page_id = None
        self._page = None
        self._dirty = False
        self._ref_bit = False
        self._pin_count = 0

    def is_evictable(self) -> bool:
        return self._pin_count == 0 and self._page_id is not None

    def __repr__(self) -> str:
        return (
            f"BufferFrame(id={self._frame_id}, page={self._page_id}, "
            f"pin={self._pin_count}, dirty={self._dirty}, ref={self._ref_bit})"
        )


# ---------------------------------------------------------------------------
# PinnedPage — RAII guard returned to callers
# ---------------------------------------------------------------------------

class PinnedPage:
    """Context manager that holds a pin on a buffer frame.

    Usage
    -----
    >>> with pool.fetch_page(page_id) as ppage:
    ...     ppage.page.write_tuple_with_length(b"hello")
    ...     ppage.mark_dirty()
    # frame is automatically unpinned on exit
    """

    def __init__(self, frame: BufferFrame, pool: "BufferPoolManager") -> None:
        self._frame = frame
        self._pool = pool
        self._dirty = False

    @property
    def page(self) -> Page:
        return self._frame.page  # type: ignore[return-value]

    def mark_dirty(self) -> None:
        self._dirty = True

    def __enter__(self) -> "PinnedPage":
        return self

    def __exit__(self, *_) -> None:
        self._pool.unpin_page(self._frame, dirty=self._dirty)


# ---------------------------------------------------------------------------
# BufferPoolManager
# ---------------------------------------------------------------------------

class BufferPoolManager:
    """Manages a fixed set of buffer frames, evicting via Clock-Sweep.

    All page access by query executors must go through fetch_page / unpin_page.
    Direct disk I/O is forbidden at higher layers.

    Parameters
    ----------
    memory_manager : MemoryManager
        Source of pre-allocated page-frame slabs.
    storage_manager : StorageManager
        Used to read/write pages on disk.
    num_frames : int
        Number of buffer frames (i.e., pool size in pages).
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        storage_manager: StorageManager,
        num_frames: int = 256,
    ) -> None:
        self._mm = memory_manager
        self._sm = storage_manager
        self._num_frames = num_frames
        self._lock = threading.Lock()

        # Allocate all frames upfront (fail-fast if budget is exceeded)
        self._frames: list[BufferFrame] = []
        for i in range(num_frames):
            block = memory_manager.alloc_page_frame(f"frame:{i}")
            self._frames.append(BufferFrame(i, block))

        # Page-id → frame lookup (the "frame table")
        self._page_table: Dict[PageId, BufferFrame] = {}

        # Clock hand for eviction
        self._clock_hand: int = 0

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_page(self, page_id: PageId) -> PinnedPage:
        """Return a pinned page frame, loading from disk on a cache miss."""
        with self._lock:
            frame = self._page_table.get(page_id)
            if frame is not None:
                frame.pin()
                self._hits += 1
                return PinnedPage(frame, self)

            # Cache miss — find a victim frame
            self._misses += 1
            victim = self._find_victim()
            if victim.page_id is not None:
                self._flush_frame(victim)
                del self._page_table[victim.page_id]
                victim.evict()
                self._evictions += 1

            # Load page from disk
            raw = self._sm.read_page(page_id.relation, page_id.page_no)
            victim.load(page_id, raw)
            self._page_table[page_id] = victim
            return PinnedPage(victim, self)

    def new_page(self, relation: str) -> PinnedPage:
        """Allocate a brand-new page on disk and return it pinned."""
        with self._lock:
            page_no = self._sm.alloc_page(relation)
            page_id = PageId(relation, page_no)
            victim = self._find_victim()
            if victim.page_id is not None:
                self._flush_frame(victim)
                del self._page_table[victim.page_id]
                victim.evict()
                self._evictions += 1
            raw = self._sm.read_page(relation, page_no)
            victim.load(page_id, raw)
            self._page_table[page_id] = victim
            return PinnedPage(victim, self)

    def unpin_page(self, frame: BufferFrame, dirty: bool = False) -> None:
        with self._lock:
            frame.unpin(dirty=dirty)

    def flush_all(self) -> int:
        """Write all dirty pages to disk.  Returns number of pages flushed."""
        count = 0
        with self._lock:
            for frame in self._frames:
                if frame.dirty and frame.page_id is not None:
                    self._flush_frame(frame)
                    count += 1
        return count

    def flush_relation(self, relation: str) -> int:
        """Flush all dirty pages belonging to a specific relation."""
        count = 0
        with self._lock:
            for pid, frame in list(self._page_table.items()):
                if pid.relation == relation and frame.dirty:
                    self._flush_frame(frame)
                    count += 1
        return count

    def invalidate_relation(self, relation: str) -> None:
        """Remove all frames belonging to a dropped relation."""
        with self._lock:
            to_remove = [pid for pid in self._page_table if pid.relation == relation]
            for pid in to_remove:
                frame = self._page_table.pop(pid)
                frame.evict()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def remaining_memory(self) -> int:
        """Remaining buffer pool memory in bytes (unpinned, non-dirty frames)."""
        with self._lock:
            free_frames = sum(
                1 for f in self._frames
                if f.page_id is None or (not f.dirty and f.pin_count == 0)
            )
        from llmdb.core.storage import PAGE_SIZE
        return free_frames * PAGE_SIZE

    def stats(self) -> Dict[str, object]:
        with self._lock:
            used = len(self._page_table)
        return {
            "num_frames": self._num_frames,
            "used_frames": used,
            "free_frames": self._num_frames - used,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_ratio": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0.0
            ),
            "remaining_bytes": self.remaining_memory,
        }

    # ------------------------------------------------------------------
    # Internal eviction (Clock-Sweep)
    # ------------------------------------------------------------------

    def _find_victim(self) -> BufferFrame:
        """Clock-sweep: advance hand until a frame with ref_bit=False is found."""
        for _ in range(2 * self._num_frames):
            frame = self._frames[self._clock_hand]
            self._clock_hand = (self._clock_hand + 1) % self._num_frames
            if frame.page_id is None:
                return frame  # free frame — use immediately
            if not frame.is_evictable():
                continue
            if frame.ref_bit:
                frame.clear_ref()
                continue
            return frame
        raise MemoryError("Buffer pool full — all frames are pinned")

    def _flush_frame(self, frame: BufferFrame) -> None:
        """Write a dirty frame to disk (must be called with self._lock held)."""
        if frame.dirty and frame.page_id is not None and frame.page is not None:
            lsn = self._sm.write_page(
                frame.page_id.relation,
                frame.page_id.page_no,
                frame.page.data,
            )
            frame.page.lsn = lsn
            frame._dirty = False  # noqa: SLF001
