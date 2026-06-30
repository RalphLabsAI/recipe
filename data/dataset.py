"""
Token-shard dataset for canonical training.

Each shard is a flat binary file of uint16 token ids (GPT-2 BPE fits in 16
bits). The dataset opens shards via memory-mapping for cheap random access,
and yields sequences of fixed length under a deterministic seed-keyed index
so the canonical training run can be re-executed bit-for-bit at the data
level (audit reproducibility).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def load_shard(path: Path | str) -> np.ndarray:
    """Memory-map a shard as uint16 array."""
    return np.memmap(path, dtype=np.uint16, mode="r")


class TokenShardDataset:
    """
    Deterministic packed-sequence iterator over a manifest's shards.

    The sequence at index i is a length-(seq_len+1) view starting at offset
    f(seed, i) inside the concatenated token stream. Targets are inputs
    shifted by one; we return tensors of length seq_len each.

    The deterministic data order is critical for audit reproducibility: a
    miner declares the seed, and the validator re-derives the exact same
    sequence of training examples on audit.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        base_dir: Path | str,
        seq_len: int,
        seed: int,
        max_open_shards: int = 128,
    ):
        from .manifest import DataManifest, verify_manifest

        self.manifest = DataManifest.from_path(manifest_path)
        self._base_dir = Path(base_dir)
        bad = verify_manifest(self.manifest, self._base_dir)
        if bad:
            raise ValueError(f"manifest verification failed: {bad}")
        self._cum = np.cumsum([0] + [s.n_tokens for s in self.manifest.shards])
        self._total = int(self._cum[-1])
        self._shard_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._max_open_shards = max(1, max_open_shards)
        self.seq_len = seq_len
        self.seed = seed
        if self._total < seq_len + 1:
            raise ValueError(f"not enough tokens ({self._total}) for seq_len {seq_len}")

    @property
    def total_tokens(self) -> int:
        return self._total

    def __len__(self) -> int:
        # Effectively unbounded; we report the number of non-overlapping windows
        # for sizing purposes, but indexing wraps modulo total_tokens.
        return max(1, self._total // (self.seq_len + 1))

    @staticmethod
    def _close_shard(shard: np.ndarray) -> None:
        mmap_obj = getattr(shard, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def close(self) -> None:
        """Close any cached shard memmaps."""
        while self._shard_cache:
            _, shard = self._shard_cache.popitem()
            self._close_shard(shard)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _shard(self, shard_idx: int) -> np.ndarray:
        shard = self._shard_cache.get(shard_idx)
        if shard is not None:
            self._shard_cache.move_to_end(shard_idx)
            return shard

        if len(self._shard_cache) >= self._max_open_shards:
            _, old = self._shard_cache.popitem(last=False)
            self._close_shard(old)

        entry = self.manifest.shards[shard_idx]
        shard = load_shard(self._base_dir / entry.relpath)
        self._shard_cache[shard_idx] = shard
        return shard

    def _global_token(self, global_idx: int) -> int:
        """Return token at global token index."""
        # Locate shard.
        shard_idx = int(np.searchsorted(self._cum, global_idx, side="right") - 1)
        within = global_idx - int(self._cum[shard_idx])
        return int(self._shard(shard_idx)[within])

    def _read_range(self, start: int, length: int) -> np.ndarray:
        """Read a contiguous range of `length` tokens starting at global `start`,
        wrapping around the total token stream as needed."""
        out = np.empty(length, dtype=np.uint16)
        filled = 0
        cursor = start % self._total
        while filled < length:
            shard_idx = int(np.searchsorted(self._cum, cursor, side="right") - 1)
            shard = self._shard(shard_idx)
            within = cursor - int(self._cum[shard_idx])
            take = min(length - filled, len(shard) - within, self._total - cursor)
            out[filled : filled + take] = shard[within : within + take]
            filled += take
            cursor = (cursor + take) % self._total
        return out

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, target_ids) for training step `step`.

        Offsets are derived deterministically from (seed, step) so two runs
        with the same (manifest, seed, seq_len) consume the exact same
        sequence of training examples in the exact same order.
        """
        # Deterministic PRNG keyed by (seed, step).
        rng = np.random.default_rng(np.array([self.seed, step], dtype=np.uint64))
        start = int(rng.integers(0, self._total))
        chunk = self._read_range(start, self.seq_len + 1)
        ids = torch.from_numpy(chunk.astype(np.int64))
        return ids[:-1], ids[1:]

    def get_batch(
        self,
        step: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a batch of (B, T) input + target tensors at the given step."""
        rng = np.random.default_rng(np.array([self.seed, step], dtype=np.uint64))
        starts = rng.integers(0, self._total, size=batch_size)
        inputs = np.empty((batch_size, self.seq_len), dtype=np.int64)
        targets = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for b, s in enumerate(starts):
            chunk = self._read_range(int(s), self.seq_len + 1).astype(np.int64)
            inputs[b] = chunk[:-1]
            targets[b] = chunk[1:]
        return torch.from_numpy(inputs), torch.from_numpy(targets)
