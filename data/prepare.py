"""
Prepare a tokenized data shard from a text source.

Phase 0 (CPU smoke tests):
    python -m data.prepare --source synthetic --out data/shards/ --shard-tokens 50000 --total-tokens 200000

Phase 0.5 (H100 real training):
    python -m data.prepare --source fineweb-edu --out data/shards/ \\
        --shard-tokens 10000000 --total-tokens 1000000000 \\
        --eval-tokens 5000000 --eval-out eval/private/

This will:
  1. Stream and tokenize ~1B tokens from FineWeb-Edu (sample-10BT subset).
  2. Hold out eval tokens first (stream-first) so train shards never overlap eval.
  3. Write training shards to data/shards/.
  4. Write a held-out eval shard to eval/private/active_tokens.bin for val_bpb.
  5. Build content-addressed manifest at data/data_manifest.json.

Requires `datasets` package: pip install 'ralph-subnet[data]'
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.manifest import build_manifest, shard_hash
from data.tokenizer import EOT_TOKEN, get_tokenizer


def synthetic_stream(seed: int = 1337):
    """A small deterministic text corpus. Not real English — just stable bytes
    so the model has something to fit. Use only for CPU smoke tests."""
    rng = random.Random(seed)
    words = [
        "the", "cat", "sat", "on", "the", "mat", "and", "looked", "around",
        "quietly", "while", "rain", "tapped", "the", "tin", "roof",
        "Ralph", "validates", "training", "recipes", "openly",
        "every", "epoch", "the", "container", "attests", "what", "it", "ran",
        "miners", "search", "patches", "validators", "score", "checkpoints",
    ]
    while True:
        sent_len = rng.randint(6, 18)
        yield " ".join(rng.choice(words) for _ in range(sent_len)) + "."


def fineweb_edu_stream():  # pragma: no cover - exercised only with `datasets` installed
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for row in ds:
        yield row["text"]


def _flush_shard(out_dir: Path, shard_idx: int, buf: list[int]) -> Path:
    shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
    np.array(buf, dtype=np.uint16).tofile(shard_path)
    return shard_path


def _write_eval_sidecar(eval_path: Path, n_tokens: int) -> None:
    """Record eval shard hash beside the train manifest (does not alter manifest schema)."""
    sidecar = eval_path.parent / "eval_sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "eval_relpath": str(eval_path.name),
                "n_tokens": n_tokens,
                "sha256": shard_hash(eval_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def tokenize_into_shards(
    out_dir: Path,
    shard_tokens: int,
    total_tokens: int,
    source: str = "synthetic",
    seed: int = 1337,
    eval_tokens: int = 0,
    eval_out: Path | None = None,
    min_doc_chars: int = 0,
) -> tuple[list[Path], Path | None]:
    """Returns (train_shard_paths, eval_shard_path_or_None).

    Eval tokens are collected stream-first from the beginning of the corpus so
    train shards never share bytes with the held-out eval set.
    """
    import time as _time

    tok = get_tokenizer()
    stream = synthetic_stream(seed) if source == "synthetic" else fineweb_edu_stream()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    eval_buf: list[int] = []
    train_buf: list[int] = []
    train_written = 0
    shard_idx = 0
    docs = 0
    skipped = 0
    t0 = _time.time()
    target = total_tokens + eval_tokens

    def _progress(label: str) -> None:
        elapsed = _time.time() - t0
        written = len(eval_buf) + train_written + len(train_buf)
        rate = written / max(elapsed, 0.01)
        pct = 100 * written / target
        print(
            f"\r  [{pct:5.1f}%] {written / 1e6:.1f}M / {target / 1e6:.0f}M tokens | "
            f"{shard_idx} shards | {docs:,} docs | {skipped:,} skipped | "
            f"{label} | {rate / 1e6:.2f}M tok/s",
            end="",
            flush=True,
        )

    for text in stream:
        if not text or len(text.strip()) < min_doc_chars:
            skipped += 1
            continue
        ids = tok.encode_ordinary(text)
        ids.append(EOT_TOKEN)
        docs += 1
        cursor = 0
        while cursor < len(ids):
            if len(eval_buf) < eval_tokens:
                take = min(len(ids) - cursor, eval_tokens - len(eval_buf))
                eval_buf.extend(ids[cursor : cursor + take])
                cursor += take
                _progress("eval")
                if len(eval_buf) >= eval_tokens and train_written >= total_tokens:
                    print()
                    break
                continue

            if train_written >= total_tokens:
                break

            take = min(len(ids) - cursor, shard_tokens - len(train_buf))
            train_buf.extend(ids[cursor : cursor + take])
            cursor += take
            _progress("train")

            if len(train_buf) >= shard_tokens:
                paths.append(_flush_shard(out_dir, shard_idx, train_buf[:shard_tokens]))
                shard_idx += 1
                train_written += shard_tokens
                train_buf = train_buf[shard_tokens:]
                if train_written >= total_tokens:
                    print()
                    break

        if len(eval_buf) >= eval_tokens and train_written >= total_tokens:
            break

    if train_buf and train_written < total_tokens:
        remaining = total_tokens - train_written
        chunk = train_buf[:remaining]
        if chunk:
            paths.append(_flush_shard(out_dir, shard_idx, chunk))
            train_written += len(chunk)
    print()

    eval_path = None
    if eval_tokens > 0 and eval_out is not None:
        eval_out.mkdir(parents=True, exist_ok=True)
        eval_path = eval_out / "active_tokens.bin"
        if len(eval_buf) < eval_tokens:
            raise RuntimeError(
                f"stream exhausted before collecting {eval_tokens:,} eval tokens "
                f"(got {len(eval_buf):,}); try lowering --eval-tokens or --min-doc-chars"
            )
        np.array(eval_buf[:eval_tokens], dtype=np.uint16).tofile(eval_path)
        _write_eval_sidecar(eval_path, eval_tokens)
        print(f"  eval shard (stream-first): {eval_path} ({eval_tokens:,} tokens)")

    return paths, eval_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "fineweb-edu"], default="synthetic")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-tokens", type=int, default=100_000)
    p.add_argument("--total-tokens", type=int, default=500_000)
    p.add_argument("--eval-tokens", type=int, default=0,
                   help="Hold out this many tokens from the start of the stream for hidden eval")
    p.add_argument("--eval-out", type=Path, default=None,
                   help="Directory for held-out eval tokens (default: eval/private/)")
    p.add_argument("--min-doc-chars", type=int, default=0,
                   help="Skip documents shorter than this (after strip). Useful for FineWeb-Edu noise.")
    p.add_argument("--track", default="llm-pretraining-launch")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    eval_out = args.eval_out or (Path(__file__).resolve().parent.parent / "eval" / "private")
    paths, eval_path = tokenize_into_shards(
        args.out,
        shard_tokens=args.shard_tokens,
        total_tokens=args.total_tokens,
        source=args.source,
        seed=args.seed,
        eval_tokens=args.eval_tokens,
        eval_out=eval_out if args.eval_tokens > 0 else None,
        min_doc_chars=args.min_doc_chars,
    )
    base_dir = args.out.parent
    manifest = build_manifest(
        track=args.track,
        tokenizer="gpt2",
        vocab_size=50257,
        dtype="uint16",
        shards=paths,
        base_dir=base_dir,
    )
    manifest_path = args.manifest if args.manifest else base_dir / "data_manifest.json"
    manifest.write(manifest_path)
    print(f"wrote {len(paths)} shards, {manifest.total_tokens():,} train tokens")
    if eval_path:
        print(f"eval shard: {eval_path}")
    print(f"manifest: {manifest_path}")
    print(f"manifest hash: {manifest.manifest_hash()[:16]}…")


if __name__ == "__main__":
    main()
