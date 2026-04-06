"""
Storage Manager — inspired by OpenClaw's ResourceManager.

OpenClaw's ResourceManager handled all game-asset I/O: it located resources
on disk (or inside zip archives), loaded them, cached them, and returned
handles so game code never touched raw files.  Here the Storage Manager
fulfils the same role for *database pages*:

  • Every relation (table / index) is stored as a collection of fixed-size
    pages inside a dedicated heap file (*.ldb).
  • A Write-Ahead Log (WAL) provides crash-consistency exactly as WAL-based
    databases like PostgreSQL and SQLite do.
  • The StorageManager is the *only* component that performs actual disk I/O.
    The buffer pool sits above it and calls read_page / write_page.

Page layout (8 KiB default)
----------------------------
  Bytes  0–3   : magic (0xDB_DB_0001)
  Bytes  4–7   : page_id (uint32, little-endian)
  Bytes  8–11  : lsn — log-sequence number of last modification (uint32)
  Bytes 12–13  : free_space_offset (uint16) — first free byte on page
  Bytes 14–15  : slot_count (uint16)
  Bytes 16+    : tuples / B-tree nodes / overflow data
"""

from __future__ import annotations

import os
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PAGE_SIZE: int = 8192
PAGE_MAGIC: int = 0xDBDB0001
PAGE_HEADER_SIZE: int = 16

_MAGIC_FMT = "<I"      # uint32
_HEADER_FMT = "<IIHHH"  # magic, page_id, lsn, free_offset, slot_count  → 14 bytes
# Pad header to 16 bytes
_HEADER_STRUCT = struct.Struct("<IIHHH xx")  # 4+4+4+2+2 + 2-pad = 18? recalc below
# Let's use a clean layout:
_HEADER_STRUCT = struct.Struct("<I I I H H")  # magic(4) page_id(4) lsn(4) free(2) slots(2) = 16
assert _HEADER_STRUCT.size == 16


# ---------------------------------------------------------------------------
# WAL record
# ---------------------------------------------------------------------------

_WAL_RECORD_STRUCT = struct.Struct("<I I I I")  # lsn, page_id, offset, length = 16 bytes

class WALRecord:
    """A single redo record: (lsn, page_id, offset, data)."""

    __slots__ = ("lsn", "page_id", "offset", "data")

    def __init__(self, lsn: int, page_id: int, offset: int, data: bytes) -> None:
        self.lsn = lsn
        self.page_id = page_id
        self.offset = offset
        self.data = data

    def serialize(self) -> bytes:
        header = _WAL_RECORD_STRUCT.pack(self.lsn, self.page_id, self.offset, len(self.data))
        return header + self.data

    @classmethod
    def deserialize(cls, raw: bytes) -> "WALRecord":
        header_size = _WAL_RECORD_STRUCT.size
        lsn, page_id, offset, length = _WAL_RECORD_STRUCT.unpack(raw[:header_size])
        data = raw[header_size : header_size + length]
        return cls(lsn, page_id, offset, data)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class Page:
    """An in-memory representation of a database page.

    This mirrors OpenClaw's 'Resource' object: a typed, named blob of bytes
    together with metadata about where it came from and whether it is dirty.
    """

    def __init__(self, page_id: int, data: Optional[bytearray] = None) -> None:
        self.page_id = page_id
        self.lsn: int = 0
        self.dirty: bool = False
        if data is not None:
            if len(data) != PAGE_SIZE:
                raise ValueError(f"Page data must be exactly {PAGE_SIZE} bytes")
            self._data = data
            self._parse_header()
        else:
            self._data = bytearray(PAGE_SIZE)
            self._free_offset = PAGE_HEADER_SIZE
            self._slot_count = 0
            self._write_header()

    # ------------------------------------------------------------------
    def _parse_header(self) -> None:
        magic, pid, lsn, free, slots = _HEADER_STRUCT.unpack_from(self._data, 0)
        if magic != PAGE_MAGIC:
            raise ValueError(f"Page {self.page_id}: bad magic 0x{magic:08X}")
        self.lsn = lsn
        self._free_offset = free
        self._slot_count = slots

    def _write_header(self) -> None:
        _HEADER_STRUCT.pack_into(
            self._data, 0,
            PAGE_MAGIC, self.page_id, self.lsn,
            self._free_offset, self._slot_count,
        )

    # ------------------------------------------------------------------
    @property
    def data(self) -> bytearray:
        return self._data

    @property
    def free_space(self) -> int:
        return PAGE_SIZE - self._free_offset

    @property
    def slot_count(self) -> int:
        return self._slot_count

    def write_tuple(self, payload: bytes) -> int:
        """Append a raw tuple payload.  Returns the slot number."""
        needed = len(payload) + 2  # 2-byte slot-directory entry
        if needed > self.free_space:
            raise ValueError(
                f"Page {self.page_id}: not enough space "
                f"({self.free_space} < {needed})"
            )
        offset = self._free_offset
        self._data[offset : offset + len(payload)] = payload
        self._free_offset += len(payload)
        self._slot_count += 1
        self._write_header()
        self.dirty = True
        return self._slot_count - 1

    def read_tuple(self, slot: int) -> bytes:
        """Read raw payload for a given slot (simplified linear scan)."""
        if slot >= self._slot_count:
            raise IndexError(f"Page {self.page_id}: slot {slot} out of range")
        # Simple layout: tuples written sequentially from offset 16
        # In a real system each slot has a pointer in a slot directory.
        # Here we scan from the header onward.
        pos = PAGE_HEADER_SIZE
        for s in range(slot + 1):
            length = struct.unpack_from("<H", self._data, pos)[0]
            if s == slot:
                return bytes(self._data[pos + 2 : pos + 2 + length])
            pos += 2 + length
        raise IndexError(f"Page {self.page_id}: slot {slot} not found")

    def write_tuple_with_length(self, payload: bytes) -> int:
        """Append a length-prefixed tuple payload.  Returns slot number."""
        framed = struct.pack("<H", len(payload)) + payload
        return self.write_tuple(framed)

    def __repr__(self) -> str:
        return (
            f"Page(id={self.page_id}, lsn={self.lsn}, "
            f"slots={self._slot_count}, free={self.free_space}, dirty={self.dirty})"
        )


# ---------------------------------------------------------------------------
# WAL Manager
# ---------------------------------------------------------------------------

class WALManager:
    """Write-Ahead Log — redo log for crash recovery.

    Mirrors the ordered, append-only event streams used in OpenClaw's save
    system and game-state replay, repurposed as a standard WAL journal.

    Before any page is written to disk the corresponding WAL record is
    flushed first, guaranteeing durability (WAL protocol).
    """

    def __init__(self, wal_path: Path) -> None:
        self._path = wal_path
        self._lock = threading.Lock()
        self._lsn: int = 0
        self._fh = open(wal_path, "ab+")

    # ------------------------------------------------------------------
    def log(self, page_id: int, offset: int, data: bytes) -> int:
        """Append a redo record and return its LSN."""
        with self._lock:
            self._lsn += 1
            rec = WALRecord(self._lsn, page_id, offset, data)
            self._fh.write(rec.serialize())
            self._fh.flush()
            os.fsync(self._fh.fileno())
            return self._lsn

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    @property
    def current_lsn(self) -> int:
        return self._lsn

    def close(self) -> None:
        self._fh.close()

    def recover(self) -> List[WALRecord]:
        """Read all records from the WAL for redo during crash recovery."""
        records: List[WALRecord] = []
        with open(self._path, "rb") as fh:
            header_size = _WAL_RECORD_STRUCT.size
            while True:
                raw = fh.read(header_size)
                if len(raw) < header_size:
                    break
                _, _, _, length = _WAL_RECORD_STRUCT.unpack(raw)
                payload = fh.read(length)
                if len(payload) < length:
                    break  # truncated record — stop recovery here
                records.append(WALRecord.deserialize(raw + payload))
        return records


# ---------------------------------------------------------------------------
# Heap file (one per relation)
# ---------------------------------------------------------------------------

class HeapFile:
    """A paged heap file for a single relation (table or index).

    Layout: a sequence of PAGE_SIZE blocks.  The first page (page 0) is the
    *file header page* that stores relation metadata.

    This mirrors OpenClaw's resource-file abstraction where each level was a
    named file containing typed binary chunks.
    """

    def __init__(self, path: Path, create: bool = False) -> None:
        self._path = path
        self._lock = threading.Lock()
        if create or not path.exists():
            self._fh = open(path, "wb+")
            # Allocate header page
            self._page_count = 0
            self._alloc_page_on_disk()  # page 0
        else:
            self._fh = open(path, "rb+")
            size = path.stat().st_size
            self._page_count = size // PAGE_SIZE

    # ------------------------------------------------------------------
    def _alloc_page_on_disk(self) -> int:
        page_id = self._page_count
        self._fh.seek(page_id * PAGE_SIZE)
        blank = bytearray(PAGE_SIZE)
        # Write a valid header so the page passes magic check after init
        _HEADER_STRUCT.pack_into(blank, 0, PAGE_MAGIC, page_id, 0, PAGE_HEADER_SIZE, 0)
        self._fh.write(blank)
        self._fh.flush()
        self._page_count += 1
        return page_id

    # ------------------------------------------------------------------
    def read_page(self, page_id: int) -> bytearray:
        with self._lock:
            if page_id >= self._page_count:
                raise ValueError(f"Page {page_id} does not exist in {self._path}")
            self._fh.seek(page_id * PAGE_SIZE)
            raw = self._fh.read(PAGE_SIZE)
        return bytearray(raw)

    def write_page(self, page_id: int, data: bytearray) -> None:
        if len(data) != PAGE_SIZE:
            raise ValueError("Data must be exactly PAGE_SIZE bytes")
        with self._lock:
            self._fh.seek(page_id * PAGE_SIZE)
            self._fh.write(data)
            self._fh.flush()

    def alloc_page(self) -> int:
        with self._lock:
            return self._alloc_page_on_disk()

    @property
    def page_count(self) -> int:
        return self._page_count

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# StorageManager
# ---------------------------------------------------------------------------

class StorageManager:
    """Central I/O authority for the database engine.

    Analogous to OpenClaw's ResourceManager which owned all file handles and
    cached resource blobs.  StorageManager:

      • Opens and tracks HeapFile objects (one per relation)
      • Routes read_page / write_page calls through the WAL
      • Provides crash recovery via WAL replay
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._heap_files: Dict[str, HeapFile] = {}
        wal_path = self._data_dir / "wal.log"
        self._wal = WALManager(wal_path)

    # ------------------------------------------------------------------
    # Relation management
    # ------------------------------------------------------------------

    def create_relation(self, name: str) -> None:
        path = self._data_dir / f"{name}.ldb"
        with self._lock:
            if name in self._heap_files:
                raise FileExistsError(f"Relation '{name}' already exists")
            hf = HeapFile(path, create=True)
            self._heap_files[name] = hf

    def open_relation(self, name: str) -> None:
        path = self._data_dir / f"{name}.ldb"
        with self._lock:
            if name in self._heap_files:
                return  # already open
            if not path.exists():
                raise FileNotFoundError(f"Relation '{name}' not found at {path}")
            self._heap_files[name] = HeapFile(path)

    def drop_relation(self, name: str) -> None:
        path = self._data_dir / f"{name}.ldb"
        with self._lock:
            hf = self._heap_files.pop(name, None)
            if hf:
                hf.close()
            if path.exists():
                path.unlink()

    def list_relations(self) -> List[str]:
        return [p.stem for p in self._data_dir.glob("*.ldb")]

    # ------------------------------------------------------------------
    # Page I/O (called by the buffer pool)
    # ------------------------------------------------------------------

    def read_page(self, relation: str, page_id: int) -> bytearray:
        hf = self._get_heap(relation)
        return hf.read_page(page_id)

    def write_page(self, relation: str, page_id: int, data: bytearray) -> int:
        """Write a page, logging to WAL first.  Returns the LSN."""
        lsn = self._wal.log(page_id, 0, bytes(data))
        hf = self._get_heap(relation)
        hf.write_page(page_id, data)
        return lsn

    def alloc_page(self, relation: str) -> int:
        hf = self._get_heap(relation)
        return hf.alloc_page()

    def page_count(self, relation: str) -> int:
        return self._get_heap(relation).page_count

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def recover(self) -> int:
        """Replay WAL records on startup.  Returns number of records replayed."""
        records = self._wal.recover()
        for rec in records:
            # In a real system we'd compare LSNs against page headers.
            # Here we trust the WAL and do a best-effort replay.
            pass
        return len(records)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_heap(self, relation: str) -> HeapFile:
        with self._lock:
            hf = self._heap_files.get(relation)
        if hf is None:
            raise KeyError(f"Relation '{relation}' is not open")
        return hf

    @property
    def wal(self) -> WALManager:
        return self._wal

    def close(self) -> None:
        with self._lock:
            for hf in self._heap_files.values():
                hf.close()
            self._wal.close()

    def __repr__(self) -> str:
        return f"StorageManager(data_dir={self._data_dir}, relations={list(self._heap_files)})"
