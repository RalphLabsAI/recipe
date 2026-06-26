"""
Canonical training loop — MUON-VS VARIANT (Variance-Adaptive Muon, arXiv 2601.14603).

Fixes our earlier broken Muon (LR was 0.02; correct is 6e-4 = AdamW LR per the
paper). Adds the variance buffer Γ and the variance-scaling
M̄ = M̃ / (sqrt(Γ̂) + ε) before Newton-Schulz. Zero new hyperparameters (reuses β).
MuonVS on 2D hidden weights; AdamW on embeddings/lm_head/norms (embeddings
excluded from weight decay). Schedule = WSD (matches our clean stack).
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import TokenShardDataset
from model import RalphBase, RalphConfig


@dataclass
class TrainConfig:
    vocab_size: int = 50257
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64
    ffn_mult: float = 8 / 3
    max_seq_len: int = 1024
    seq_len: int = 256
    batch_size: int = 16
    micro_batch_size: int = 16
    total_steps: int = 200
    warmup_steps: int = 20
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    muon_beta: float = 0.95
    muon_ns_steps: int = 5
    manifest_path: str = "data/data_manifest.json"
    data_base_dir: str = "data"
    data_seed: int = 1337
    init_seed: int = 1337
    use_bf16: bool = True
    log_every: int = 10

    @property
    def grad_accum_steps(self) -> int:
        assert self.batch_size % self.micro_batch_size == 0
        return self.batch_size // self.micro_batch_size


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    # WSD: warmup -> hold peak -> linear cooldown over final 20%
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / max(1, cfg.warmup_steps)
    decay_start = int(cfg.total_steps * 0.8)
    if step < decay_start:
        return cfg.max_lr
    dp = (step - decay_start) / max(1, cfg.total_steps - decay_start)
    dp = min(1.0, max(0.0, dp))
    return cfg.max_lr + (cfg.min_lr - cfg.max_lr) * dp


def build_model(cfg: TrainConfig) -> RalphBase:
    return RalphBase(RalphConfig(
        vocab_size=cfg.vocab_size, dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        head_dim=cfg.head_dim, ffn_mult=cfg.ffn_mult, max_seq_len=cfg.max_seq_len))


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T; transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MuonVS(torch.optim.Optimizer):
    """Variance-Adaptive Muon (2601.14603). Per 2D weight W."""
    def __init__(self, params, lr=6e-4, beta=0.95, weight_decay=0.0, ns_steps=5, eps=1e-8):
        super().__init__(params, dict(lr=lr, beta=beta, weight_decay=weight_decay, ns_steps=ns_steps, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, beta, wd, ns, eps = (group["lr"], group["beta"], group["weight_decay"],
                                     group["ns_steps"], group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g); st["v"] = torch.zeros_like(g); st["t"] = 0
                m, v = st["m"], st["v"]
                st["t"] += 1; t = st["t"]
                v.mul_(beta).add_((g - m) ** 2, alpha=1 - beta)   # variance uses OLD m
                m.mul_(beta).add_(g, alpha=1 - beta)
                bc = 1 - beta ** t
                mhat = m / bc
                vhat = v / bc
                mtil = g + (beta / (1 - beta)) * mhat              # Nesterov lookahead
                mbar = mtil / (vhat.sqrt() + eps)                  # variance scaling
                o = _zeropower_via_newtonschulz5(mbar, steps=ns)
                scale = 0.2 * (max(p.size(0), p.size(1)) ** 0.5)   # RMS-match
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(o.to(p.dtype), alpha=-lr * scale)
        return None


class CombinedOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]
    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)
    @torch.no_grad()
    def step(self, closure=None):
        for o in self.optimizers:
            o.step()


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = ("tok_embed" in n) or ("lm_head" in n)
        if p.dim() >= 2 and not is_embed:
            muon_params.append(p)
        else:
            adamw_nodecay.append(p)   # embeddings + norms/biases: no weight decay (nowdembed)
    adamw = torch.optim.AdamW(
        [{"params": adamw_nodecay, "weight_decay": 0.0, "lr_scale": 1.0}],
        lr=cfg.max_lr, betas=(cfg.beta1, cfg.beta2))
    muon = MuonVS(muon_params, lr=cfg.max_lr, beta=cfg.muon_beta, weight_decay=cfg.weight_decay,
                  ns_steps=cfg.muon_ns_steps)
    for g in muon.param_groups:
        g["lr_scale"] = 1.0
    return CombinedOptimizer([muon, adamw])


def train(cfg: TrainConfig, out_dir: Path, use_wandb: bool = False) -> dict:
    set_determinism(cfg.init_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    ds = TokenShardDataset(cfg.manifest_path, cfg.data_base_dir, cfg.seq_len, cfg.data_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "training_log.jsonl").open("w")
    use_amp = cfg.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler(device.type, enabled=False) if device.type == "cuda" else None
    n_params = model.num_parameters()
    print(f"[train] device={device} params={n_params:,} optimizer=MuonVS(2D)+AdamW(embed/norm) lr={cfg.max_lr}")
    start = time.time(); tokens_seen = 0; last_loss = float("nan")
    for step in range(cfg.total_steps):
        lr = cosine_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        step_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for accum in range(cfg.grad_accum_steps):
            sub_step = step * cfg.grad_accum_steps + accum
            inp, tgt = ds.get_batch(sub_step, cfg.micro_batch_size)
            inp = inp.to(device, non_blocking=True); tgt = tgt.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(inp, targets=tgt)
            (loss / cfg.grad_accum_steps).backward()
            step_loss += loss.item() / cfg.grad_accum_steps
            tokens_seen += cfg.micro_batch_size * cfg.seq_len
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        optimizer.step()
        last_loss = step_loss
        elapsed = time.time() - start
        entry = {"step": step, "loss": step_loss, "lr": lr, "grad_norm": grad_norm,
                 "tokens_seen": tokens_seen, "tokens_per_sec": tokens_seen / max(elapsed, 1e-6),
                 "elapsed_s": elapsed}
        if step % cfg.log_every == 0 or step == cfg.total_steps - 1:
            log_f.write(json.dumps(entry) + "\n")
            print(f"[step {step:4d}/{cfg.total_steps}] loss={step_loss:.4f} lr={lr:.2e} |g|={grad_norm:.2f}")
    log_f.close()
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out_dir / "checkpoint.pt")
    summary = {"steps": cfg.total_steps, "final_loss": last_loss, "tokens_seen": tokens_seen,
               "wall_clock_s": time.time() - start, "n_params": n_params,
               "n_params_no_embed": model.num_parameters(exclude_embeddings=True),
               "manifest_hash": ds.manifest.manifest_hash(), "device": str(device),
               "precision": "bf16" if use_amp else "fp32", "config": asdict(cfg)}
    (out_dir / "final_state.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done. final loss={last_loss:.4f} wall={summary['wall_clock_s']:.1f}s")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()
    cfg = TrainConfig()
    if args.config and args.config.exists():
        for k, v in json.loads(args.config.read_text()).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.manifest is not None:
        cfg.manifest_path = str(args.manifest)
    if args.seed is not None:
        cfg.init_seed = args.seed; cfg.data_seed = args.seed
    train(cfg, args.out_dir, use_wandb=args.wandb)


if __name__ == "__main__":
    main()
