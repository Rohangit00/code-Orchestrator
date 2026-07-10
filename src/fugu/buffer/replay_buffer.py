"""Thread-safe replay buffer with compressed on-disk persistence.

Transitions are serialised via :meth:`Transition.to_dict`, packed with
*msgpack*, and compressed with *zstandard* (level 3) before being stored
in an in-memory list.  The buffer supports random sampling, disk
save/load, and automatic eviction when the configured capacity is reached.
"""

from __future__ import annotations

import logging
import random
import struct
import threading
from pathlib import Path
from typing import Iterator

import msgpack
import zstandard as zstd

from fugu.core.state import Transition

logger = logging.getLogger(__name__)

# File format magic bytes to identify buffer files.
_MAGIC = b"FUGU_RB\x00"
_HEADER_VERSION = 1


class ReplayBuffer:
    """Compressed, thread-safe replay buffer for MDP transitions.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to store.  When full, the oldest
        transition is evicted (FIFO).
    storage_dir:
        Default directory for :meth:`save` / :meth:`load` when no explicit
        path is given.
    max_size_mb:
        Soft upper bound on in-memory size.  Not strictly enforced — used
        by :meth:`disk_usage_mb` to report usage relative to budget.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        storage_dir: str = "data/buffer",
        max_size_mb: int = 512,
    ) -> None:
        self._capacity = capacity
        self._storage_dir = Path(storage_dir)
        self._max_size_mb = max_size_mb
        self._buffer: list[bytes] = []  # list of compressed transition bytes
        self._lock = threading.Lock()
        self._compressor = zstd.ZstdCompressor(level=3)
        self._decompressor = zstd.ZstdDecompressor()
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of stored transitions."""
        return self._capacity

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, transition: Transition) -> None:
        """Serialize, compress, and store a single transition.

        Thread-safe.  If the buffer is at capacity, the oldest entry is
        silently evicted.
        """
        data = transition.to_dict()
        packed = msgpack.packb(data, use_bin_type=True)
        compressed = self._compressor.compress(packed)

        with self._lock:
            if len(self._buffer) >= self._capacity:
                self._buffer.pop(0)
            self._buffer.append(compressed)

    def add_episode(self, transitions: list[Transition]) -> None:
        """Add all transitions from an episode in order."""
        for t in transitions:
            self.add(t)

    def sample(self, batch_size: int) -> list[Transition]:
        """Return a uniformly random sample of transitions.

        Each sampled entry is decompressed and reconstructed into a full
        :class:`Transition` object.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A list of reconstructed transitions.

        Raises:
            ValueError: If *batch_size* exceeds the buffer length.
        """
        with self._lock:
            if batch_size > len(self._buffer):
                raise ValueError(
                    f"Cannot sample {batch_size} transitions from a buffer "
                    f"of size {len(self._buffer)}."
                )
            sampled_bytes = random.sample(self._buffer, batch_size)

        return [self._decompress(b) for b in sampled_bytes]

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Persist the buffer to a single file on disk.

        File format::

            8 bytes  — magic ``FUGU_RB\\x00``
            4 bytes  — header version (uint32 LE)
            4 bytes  — number of entries (uint32 LE)
            For each entry:
                4 bytes  — entry length in bytes (uint32 LE)
                N bytes  — compressed transition data

        Args:
            path: Target file.  Defaults to ``<storage_dir>/buffer.bin``.
        """
        dest = Path(path) if path is not None else self._storage_dir / "buffer.bin"
        dest.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            snapshot = list(self._buffer)

        with open(dest, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<I", _HEADER_VERSION))
            f.write(struct.pack("<I", len(snapshot)))
            for entry in snapshot:
                f.write(struct.pack("<I", len(entry)))
                f.write(entry)

        logger.info(
            "Saved %d transitions to %s (%.2f MB)",
            len(snapshot),
            dest,
            dest.stat().st_size / (1024 * 1024),
        )

    def load(self, path: str | Path | None = None) -> None:
        """Load transitions from a previously saved buffer file.

        Existing in-memory data is **replaced** by the loaded content.

        Args:
            path: Source file.  Defaults to ``<storage_dir>/buffer.bin``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file header is invalid.
        """
        src = Path(path) if path is not None else self._storage_dir / "buffer.bin"

        with open(src, "rb") as f:
            magic = f.read(len(_MAGIC))
            if magic != _MAGIC:
                raise ValueError(
                    f"Invalid buffer file: expected magic {_MAGIC!r}, got {magic!r}"
                )

            (version,) = struct.unpack("<I", f.read(4))
            if version != _HEADER_VERSION:
                raise ValueError(
                    f"Unsupported buffer version {version}; expected {_HEADER_VERSION}"
                )

            (count,) = struct.unpack("<I", f.read(4))
            entries: list[bytes] = []
            for _ in range(count):
                (length,) = struct.unpack("<I", f.read(4))
                entries.append(f.read(length))

        with self._lock:
            self._buffer = entries

        logger.info("Loaded %d transitions from %s", count, src)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def disk_usage_mb(self) -> float:
        """Approximate in-memory size of all compressed entries in megabytes."""
        with self._lock:
            total_bytes = sum(len(entry) for entry in self._buffer)
        return total_bytes / (1024 * 1024)

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all transitions from the buffer."""
        with self._lock:
            self._buffer.clear()

    def get_all(self) -> list[Transition]:
        """Decompress and return every stored transition."""
        with self._lock:
            snapshot = list(self._buffer)
        return [self._decompress(b) for b in snapshot]

    def __iter__(self) -> Iterator[Transition]:
        """Iterate over all stored transitions (decompresses lazily)."""
        with self._lock:
            snapshot = list(self._buffer)
        for b in snapshot:
            yield self._decompress(b)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(len={len(self._buffer)}, "
            f"capacity={self._capacity}, "
            f"usage_mb={self.disk_usage_mb():.2f})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decompress(self, data: bytes) -> Transition:
        """Decompress and reconstruct a single transition from raw bytes."""
        decompressed = self._decompressor.decompress(data)
        unpacked = msgpack.unpackb(decompressed, raw=False)
        return Transition.from_dict(unpacked)
