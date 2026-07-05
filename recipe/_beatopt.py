"""Beat optimizer (NEW file): self-contained Muon + Newton-Schulz + decoupled
muon_weight_decay + embed_lr group. getattr-defaults so it works on canonical TrainConfig.
"""
from __future__ import annotations
import torch

def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to orthogonalize the update matrix (Muon).
    Computes G (G^T G)^(-1/2) approximately via a quintic iteration in bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)



class Muon(torch.optim.Optimizer):
    """Momentum orthogonalized by Newton-Schulz, for 2D hidden weight matrices.
    See Keller Jordan's modded-nanogpt. Embeddings/heads/norms use AdamW instead."""

    def __init__(self, params, lr=0.04, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(p.grad)
                upd = p.grad.add(buf, alpha=mom) if group["nesterov"] else buf
                upd = _zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                # Scale so the RMS update magnitude is ~LR-invariant to matrix shape.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd:
                    p.mul_(1.0 - lr * scale * wd)
                p.add_(upd, alpha=-lr * scale)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    """Returns a LIST of optimizers stepped together. Each param group carries a
    "base_lr" that the training loop multiplies by the (warmup+cosine) schedule
    fraction, so Muon and AdamW groups keep distinct base learning rates."""
    if cfg.optimizer == "muon":
        muon_params, embed_params, norm_params = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "tok_embed" in n or "lm_head" in n:
                embed_params.append(p)
            elif p.dim() >= 2:
                muon_params.append(p)
            else:
                norm_params.append(p)
        # DEFAULTS are OUR beat values: canonical TrainConfig drops these fields, so
        # the bundle (where scoring happens) relies on these defaults, not the config.
        muon = Muon(muon_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps, weight_decay=getattr(cfg, 'muon_weight_decay', 0.05))
        emb_lr = getattr(cfg, "embed_lr", 0.006) or cfg.max_lr
        adamw = torch.optim.AdamW(
            [
                {"params": embed_params, "weight_decay": cfg.weight_decay, "lr": emb_lr},
                {"params": norm_params, "weight_decay": 0.0, "lr": cfg.max_lr},
            ],
            lr=cfg.max_lr,
            betas=(cfg.beta1, cfg.beta2),
        )
        muon.param_groups[0]["base_lr"] = cfg.muon_lr
        adamw.param_groups[0]["base_lr"] = emb_lr
        adamw.param_groups[1]["base_lr"] = cfg.max_lr
        return [muon, adamw]

    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    adamw = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
    )
    for grp in adamw.param_groups:
        grp["base_lr"] = cfg.max_lr
    return [adamw]


