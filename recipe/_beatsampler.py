"""Offset-window sampler as a NEW file (touches nothing #1317 changed).
Subclasses the canonical TokenShardDataset and overrides get_batch: 100% per-epoch
coverage + deterministic per-epoch boundary offset. Pure fn of (seed,seq_len,step)."""
import numpy as np, torch
from data.dataset import TokenShardDataset

class BeatDataset(TokenShardDataset):
    def get_batch(self, step, batch_size):
        W = self.seq_len + 1; n_win = self._total // W
        inp = np.empty((batch_size, self.seq_len), dtype=np.int64)
        tgt = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for b in range(batch_size):
            k = step * batch_size + b; epoch, idx = divmod(k, n_win)
            if getattr(self, "_perm_epoch", None) != epoch:
                self._perm = np.random.default_rng(np.array([self.seed, 0xE90C4, epoch], dtype=np.uint64)).permutation(n_win)
                self._off = int(np.random.default_rng(np.array([self.seed, 0x0FF5E7, epoch], dtype=np.uint64)).integers(0, W))
                self._perm_epoch = epoch
            s = (self._off + int(self._perm[idx]) * W) % (self._total - W)
            chunk = self._read_range(int(s), self.seq_len + 1).astype(np.int64)
            inp[b] = chunk[:-1]; tgt[b] = chunk[1:]
        return torch.from_numpy(inp), torch.from_numpy(tgt)
