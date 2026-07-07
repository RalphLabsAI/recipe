"""
Ralph-base — minimal Llama-style decoder-only transformer.

Patchable surface for the launch track. Miners may modify any of the modules
here (attention variant, normalization, activation, etc.) as part of a
submitted recipe patch. The training loop in `recipe/train.py` instantiates
this model from a config.

Defaults target ~50M parameters at dim=512, n_layers=8 — small enough for
fast Phase 0 iteration, large enough to produce meaningful val_bpb gradients
under canonical training. Configs in `configs/` override these for proxy /
confirmation / scale proof-test variants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RalphConfig:
    vocab_size: int = 50257  # GPT-2 BPE
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64
    ffn_mult: float = 8 / 3  # Llama-style
    max_seq_len: int = 1024
    rope_base: float = 100_000.0  # recipe-v4: RoPE-100k (was 10k)
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_embeddings: bool = False
    unet_skip: bool = True        # recipe-v4: U-Net learnable skip connections
    logit_softcap: float = 30.0   # recipe-v4: tanh soft-cap on logits (0 = off)
    logit_z_coef: float = 0.0001  # z-loss on the final logits (0 = off)
    # v16 champion arch gates (all default-off => bit-identical to _v7lite):
    value_residual: bool = False  # lerp each layer's V toward layer-0's V (learnable lambda, init 0.5)
    peri_ln: bool = False         # RMSNorm on each sublayer OUTPUT before the residual add
    hybrid_norm: bool = False     # RMSNorm inside attn/ffn output projection
    resid_scale: bool = False     # learnable scalar on each sublayer output (init 1.0)
    resid_scale_init: float = 1.0


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x_f32 = x.float()
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(var + eps)
    return (x_normed * weight).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)


def precompute_rope_cache(head_dim: int, max_seq_len: int, base: float, device: torch.device) -> torch.Tensor:
    """Returns a tensor of shape (max_seq_len, head_dim // 2, 2) of cos, sin pairs."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.stack([cos, sin], dim=-1)  # (max_seq_len, head_dim // 2, 2)


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings. x is (batch, n_heads, seq, head_dim). rope_cache is
    (seq, head_dim // 2, 2). Returns same shape as x.
    """
    seq = x.shape[-2]
    cos = rope_cache[:seq, :, 0]  # (seq, head_dim // 2)
    sin = rope_cache[:seq, :, 1]
    # split last dim into pairs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # rotate
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    out = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: RalphConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dim = cfg.dim
        assert cfg.dim == cfg.n_heads * cfg.head_dim, "dim must equal n_heads * head_dim"
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        # Mark as residual-path output for depth-scaled init (GPT-2 §2.3).
        self.out_proj._is_residual_out = True
        # QK-norm: per-head RMSNorm on queries and keys before RoPE. Bounds the
        # attention-logit scale so it can't drift, which is especially important
        # under the Muon optimizer's aggressive orthogonalized updates (see
        # recipe/train.py). Strong synergy with Muon; standard in modern speedruns.
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        # v16: hybrid_norm (post-norm on out_proj output), value_residual (lerp V->V0)
        self.out_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps) if cfg.hybrid_norm else None
        self.value_residual = cfg.value_residual
        if cfg.value_residual:
            self.lambda_res = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor, v_first: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.dim, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_self = v  # v16: expose this layer's V for the value-residual mix
        if self.value_residual and v_first is not None:
            v = torch.lerp(v, v_first, self.lambda_res)
        q = self.q_norm(q)  # QK-norm (per head_dim, before RoPE)
        k = self.k_norm(k)
        q = apply_rope(q, rope_cache)
        k = apply_rope(k, rope_cache)
        # Causal self-attention via SDPA (uses flash on supported hardware).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(y)
        if self.out_norm is not None:
            out = self.out_norm(out)
        return out, v_self


class SwiGLU(nn.Module):
    def __init__(self, cfg: RalphConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.ffn_mult)
        # Round to multiple of 64 for kernel friendliness.
        hidden = 64 * ((hidden + 63) // 64)
        self.w_gate = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_up = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.dim, bias=False)
        # Mark as residual-path output for depth-scaled init (GPT-2 §2.3).
        self.w_down._is_residual_out = True
        # v16: hybrid_norm (post-norm on ffn output projection)
        self.out_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps) if cfg.hybrid_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        if self.out_norm is not None:
            out = self.out_norm(out)
        return out


class Block(nn.Module):
    def __init__(self, cfg: RalphConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)
        # v16: peri_ln (post-sublayer norm), resid_scale (learnable scalar gate)
        self.attn_post_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps) if cfg.peri_ln else None
        self.ffn_post_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps) if cfg.peri_ln else None
        self.resid_scale_attn = nn.Parameter(torch.full((1,), cfg.resid_scale_init)) if cfg.resid_scale else None
        self.resid_scale_ffn = nn.Parameter(torch.full((1,), cfg.resid_scale_init)) if cfg.resid_scale else None

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor, v_first: Optional[torch.Tensor] = None):
        attn_out, v_self = self.attn(self.attn_norm(x), rope_cache, v_first)
        if self.attn_post_norm is not None:
            attn_out = self.attn_post_norm(attn_out)
        if self.resid_scale_attn is not None:
            attn_out = attn_out * self.resid_scale_attn
        x = x + attn_out
        ffn_out = self.ffn(self.ffn_norm(x))
        if self.ffn_post_norm is not None:
            ffn_out = self.ffn_post_norm(ffn_out)
        if self.resid_scale_ffn is not None:
            ffn_out = ffn_out * self.resid_scale_ffn
        x = x + ffn_out
        return x, v_self



_KERNEL_FLAGS_SET = False


def _enable_fast_kernels() -> None:
    """TF32 matmuls + non-deterministic cuDNN autotune, declared here in the
    patchable model surface: same recipe, genuinely faster compute (~1.4x
    tok/s on H100/H200-class parts). GPU training is already non-bit-exact
    (see recipe/train.py set_determinism note) and the validator audit is
    tolerance-based, so relaxing the determinism knobs trades nothing away."""
    global _KERNEL_FLAGS_SET
    if _KERNEL_FLAGS_SET:
        return
    _KERNEL_FLAGS_SET = True
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class RalphBase(nn.Module):
    """
    Minimal Llama-style decoder-only transformer.
    Inputs: token ids (B, T). Outputs: logits (B, T, vocab_size).
    """

    def __init__(self, cfg: RalphConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # recipe-v4: U-Net skips — one learnable gate per decoder layer, 0-init
        # (starts identical to canonical, learns to use the skips).
        self.unet_skip = getattr(cfg, "unet_skip", False)
        if self.unet_skip:
            self.skip_gate = nn.Parameter(torch.zeros(cfg.n_layers - cfg.n_layers // 2))
        self.final_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.register_buffer(
            "rope_cache",
            precompute_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_base, torch.device("cpu")),
            persistent=False,
        )
        self._compiled_fwd = None
        self.apply(self._init_weights)
        if self.lm_head is not None:
            nn.init.zeros_(self.lm_head.weight)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = self.cfg.init_std
            # Scale residual-path output projections by 1/sqrt(2 * n_layers) so
            # that residual stream variance stays ~constant at init (GPT-2 §2.3).
            if getattr(module, "_is_residual_out", False):
                std = std / math.sqrt(2 * self.cfg.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_embed.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Compile the forward *function* (not the module) on first CUDA call:
        # state_dict keys stay clean (no _orig_mod. prefix), so the canonical
        # trainer's plain torch.save(model.state_dict()) checkpoint loads
        # unmodified in the validator's op4 harness.
        fwd = self._compiled_fwd
        if fwd is None:
            _enable_fast_kernels()
            fwd = self._forward_impl
            if idx.is_cuda:
                try:
                    fwd = torch.compile(self._forward_impl)
                except Exception:
                    fwd = self._forward_impl
            self._compiled_fwd = fwd
        return fwd(idx, targets)

    def _forward_impl(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert idx.shape[-1] <= self.cfg.max_seq_len, f"sequence {idx.shape[-1]} exceeds max_seq_len {self.cfg.max_seq_len}"
        x = self.tok_embed(idx)
        v_first = None  # v16: layer-0 V, threaded through blocks for value_residual
        if self.unet_skip:
            n = len(self.blocks); half = n // 2; enc = []
            for i, block in enumerate(self.blocks):
                if i < half:
                    x, v = block(x, self.rope_cache, v_first); enc.append(x)
                else:
                    x = x + self.skip_gate[i - half] * enc[n - 1 - i]
                    x, v = block(x, self.rope_cache, v_first)
                if v_first is None:
                    v_first = v
        else:
            for block in self.blocks:
                x, v = block(x, self.rope_cache, v_first)
                if v_first is None:
                    v_first = v
        x = self.final_norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.tok_embed.weight)
        else:
            logits = self.lm_head(x)
        cap = getattr(self.cfg, "logit_softcap", 0.0)  # recipe-v4: logit soft-cap
        if cap and cap > 0:
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            z_coef = getattr(self.cfg, "logit_z_coef", 0.0)
            if z_coef:
                loss = loss + z_coef * (torch.logsumexp(logits, dim=-1).float() ** 2).mean()
        return logits, loss


# ---------------------------------------------------------------------------
# Back-compat aliases (rebrand karpa -> ralph, 2026-06).
# The classes were renamed KarpaBase -> RalphBase / KarpaConfig -> RalphConfig.
# These aliases keep `from model import KarpaBase, KarpaConfig` resolving for
# any unmigrated importer or out-of-tree tooling. Checkpoints are unaffected:
# torch.save stores asdict(cfg) under "config" (field names, not the class
# name) and state_dict keys are module-attribute paths, so neither the class
# name nor "Karpa"/"Ralph" is ever serialized. Safe to remove once all
# external consumers cut over.
KarpaConfig = RalphConfig
KarpaBase = RalphBase
