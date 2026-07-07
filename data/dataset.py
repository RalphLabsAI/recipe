"""
Token-shard dataset for canonical training.

Each shard is a flat binary file of uint16 token ids (GPT-2 BPE fits in 16
bits). The dataset opens shards via memory-mapping for cheap random access,
and yields sequences of fixed length under a deterministic seed-keyed index
so the canonical training run can be re-executed bit-for-bit at the data
level (audit reproducibility).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    ):
        from .manifest import DataManifest, verify_manifest

        self.manifest = DataManifest.from_path(manifest_path)
        base = Path(base_dir)
        bad = verify_manifest(self.manifest, base)
        if bad:
            raise ValueError(f"manifest verification failed: {bad}")
        self._shards = [load_shard(base / s.relpath) for s in self.manifest.shards]
        self._cum = np.cumsum([0] + [len(s) for s in self._shards])
        self._total = int(self._cum[-1])
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

    def _global_token(self, global_idx: int) -> int:
        """Return token at global byte index."""
        # Locate shard.
        shard_idx = int(np.searchsorted(self._cum, global_idx, side="right") - 1)
        within = global_idx - int(self._cum[shard_idx])
        return int(self._shards[shard_idx][within])

    def _read_range(self, start: int, length: int) -> np.ndarray:
        """Read a contiguous range of `length` tokens starting at global `start`,
        wrapping around the total token stream as needed."""
        out = np.empty(length, dtype=np.uint16)
        filled = 0
        cursor = start % self._total
        while filled < length:
            shard_idx = int(np.searchsorted(self._cum, cursor, side="right") - 1)
            shard = self._shards[shard_idx]
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

    def _perm_window_start(self, k: int) -> int:
        """Start offset of the k-th sample under epoch-wise permutation.

        Non-overlapping windows, reshuffled each epoch with a (seed, epoch)-keyed
        permutation: every token is seen once per epoch (uniform repeat counts),
        unlike independent uniform draws which leave ~exp(-epochs) of the corpus
        unseen. Same determinism contract: order is a pure function of
        (manifest, seed, seq_len, k), so audit replay is unchanged.
        """
        n_win = self._total // (self.seq_len + 1)
        epoch, idx = divmod(k, n_win)
        if getattr(self, "_perm_epoch", None) != epoch:
            prng = np.random.default_rng(np.array([self.seed, 0xE90C4, epoch], dtype=np.uint64))
            self._perm = prng.permutation(n_win)
            self._perm_epoch = epoch
        return int(self._perm[idx]) * (self.seq_len + 1)

    def get_batch(
        self,
        step: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a batch of (B, T) input + target tensors at the given step."""
        starts = [self._perm_window_start(step * batch_size + b) for b in range(batch_size)]
        inputs = np.empty((batch_size, self.seq_len), dtype=np.int64)
        targets = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for b, s in enumerate(starts):
            chunk = self._read_range(int(s), self.seq_len + 1).astype(np.int64)
            inputs[b] = chunk[:-1]
            targets[b] = chunk[1:]
        return torch.from_numpy(inputs), torch.from_numpy(targets)
