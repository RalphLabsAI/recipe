"""Muon optimizer — port z karpathy/modded-nanogpt.

Stosuj TYLKO do 2D hidden layer params (wagi liniowe: attention QKV/out, FFN).
Embedding / lm_head / norm weights / 1D params → zostaw na AdamW.

Ref: Keller Jordan et al. (2024), github.com/KellerJordan/modded-nanogpt
"""
from __future__ import annotations

import torch


def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize G via Newton-Schulz iterations (float32).

    G is assumed to be (m, n). Works on both tall (m >= n) and wide (m < n)
    matrices by transposing as needed. The output has the same shape as G.
    """
    # Work in float32 for numerical stability regardless of input dtype
    X = G.float()
    X = X / (X.norm() + 1e-7)
    # Ensure tall matrix convention (m >= n)
    transposed = X.shape[0] < X.shape[1]
    if transposed:
        X = X.T
    # Cubic Horner recurrence — Zolotarev rational approximation coefficients
    # tuned for the spectral interval [0, 1] (from modded-nanogpt)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    return (X.T if transposed else X)


class Muon(torch.optim.Optimizer):
    """Momentum + Newton-Schulz orthogonalization.

    Designed for 2D weight matrices of hidden transformer layers:
      - Attention: qkv weight (dim, 3*dim) or split q/k/v
      - Attention: out_proj weight (dim, dim)
      - FFN: w_gate, w_up, w_down

    NOT suitable for:
      - Embedding tables (tied or untied)
      - Final LM head
      - RMSNorm / LayerNorm weights
      - Any 1D parameters or bias vectors

    Args:
        params: iterable of 2D parameters (filtered by caller)
        lr: learning rate. Muon operates on a different scale than AdamW.
            Good starting point: 0.02 (vs AdamW ~6e-4 for same model).
        momentum: Nesterov momentum coefficient (default 0.95)
        ns_steps: Newton-Schulz iteration count (default 5, sufficient)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.dim() < 2:
                    # Skip 1D params — caller should have excluded them but
                    # be defensive so a misuse doesn't silently corrupt training
                    continue

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]

                # Nesterov-style heavy-ball: buf = momentum * buf + g
                buf.mul_(momentum).add_(g)
                # Nesterov lookahead: use gradient + momentum * buf
                g_nesterov = g.add(buf, alpha=momentum)

                # Orthogonalize via Newton-Schulz (float32 internally)
                g_orth = _newton_schulz(g_nesterov, steps=ns_steps).to(p.dtype)

                # Scale update so the RMS norm of g_orth is ~1
                # (matches the "normalized" step expected by the LR heuristic)
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(g_orth, alpha=(-lr * scale))

        return loss


def partition_params_muon_adamw(model: torch.nn.Module) -> tuple[list, list, list]:
    """Split model parameters into three groups for Muon/AdamW mixed training.

    Returns:
        muon_params: 2D hidden weight matrices → Muon
        adamw_decay: embeddings / untied lm_head (2D but not hidden) → AdamW with WD
        adamw_nodecay: norms, biases, 1D params → AdamW without WD

    Usage:
        muon_p, adamw_decay_p, adamw_nodecay_p = partition_params_muon_adamw(model)
        opt_muon  = Muon(muon_p, lr=0.02)
        opt_adamw = torch.optim.AdamW([
            {"params": adamw_decay_p, "weight_decay": cfg.weight_decay},
            {"params": adamw_nodecay_p, "weight_decay": 0.0},
        ], lr=cfg.max_lr, betas=(cfg.beta1, cfg.beta2))
    """
    muon_params = []
    adamw_decay = []
    adamw_nodecay = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden_2d = (
            p.dim() >= 2
            and "tok_embed" not in name
            and "lm_head" not in name
        )
        if is_hidden_2d:
            muon_params.append(p)
        elif p.dim() >= 2:
            # Embedding matrix or untied lm_head: use AdamW with weight decay
            adamw_decay.append(p)
        else:
            # Norm scales, bias terms, 1D params: AdamW no weight decay
            adamw_nodecay.append(p)

    return muon_params, adamw_decay, adamw_nodecay
