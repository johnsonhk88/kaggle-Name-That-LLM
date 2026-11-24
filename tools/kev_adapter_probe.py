#!/usr/bin/env python3
"""Zero-shot Kev-9B on train.csv. Off-Kaggle study only — not the submit path.

Kev is a PEFT adapter (LoRA r16) plus a pointer head on frozen Qwen3.5-9B-Base. One `choice`
question over the 18 LABELS, one forward pass per row, probabilities only, no generation.

The adapter is 45.4M params and `head.pt` holds the pointer head and its fitted temperature
(T = 2.19), so this loads the 9B base once — never a second 18 GB checkpoint.

Usage
  python tools/kev_adapter_probe.py --limit 32                     # smoke
  python tools/kev_adapter_probe.py                                # Hub adapter
  python tools/kev_adapter_probe.py --model hf_cache/kev-9b --base hf_cache/qwen3.5-9b-base
  python tools/kev_adapter_probe.py --unmerged                     # keep the base reusable
  python tools/kev_adapter_probe.py --shuffle-options 4            # order-robustness check
  python tools/kev_adapter_probe.py --compare-preds outputs/d3_zeroshot_label_preds.parquet

Writes outputs/kev_zeroshot.json and outputs/kev_zeroshot_preds.parquet.

T4 note: `--unmerged` is the submit-path shape (base stays loaded, adapter swaps in). The default
merges for speed, which folds the adapter into the base weights and throws the adapter module away.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]
LABEL_INDEX = {name: i for i, name in enumerate(LABELS)}
INSTRUCTIONS = "Which of these 18 language models wrote this passage? Pick exactly one author."
LABEL_CRITERIA = {name: f"Written by {name}." for name in LABELS}
TFIDF_OOF = 0.3152249565001243


def load_kev(model: Path, base: str | None, device: str, dtype_name: str, merge: bool, temperature: float | None):
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions

    ckpt = Checkpoint(str(model))
    if base:
        # Offline / Kaggle: point the frozen base at the attached directory and drop the Hub revision pin.
        ckpt.meta.base = base
        ckpt.meta.base_revision = None
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    opts = LoadOptions(dtype=dtype, merge=merge, temperature=temperature)
    tok, model_obj = ckpt.load(device, opts)
    print(
        f"loaded Kev from {model}: base={ckpt.meta.base} T={model_obj.head.temperature:.2f} "
        f"merge={merge} dtype={dtype_name}",
        flush=True,
    )
    return tok, model_obj


def score_one(tok, model, text: str, criteria: dict[str, str], shuffle: int, rng: random.Random):
    """-> (probabilities in LABELS order, picks across permutations, passes)."""
    from kev.api import SystemOneRequest, to_record

    order = list(criteria)
    perms = [order] if shuffle <= 1 else [rng.sample(order, len(order)) for _ in range(shuffle)]
    total = np.zeros(len(LABELS), dtype=np.float64)
    picks: list[str] = []
    for perm in perms:
        request = {
            "state": text,
            "questions": {
                "model": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": {name: criteria[name] for name in perm},
                }
            },
        }
        rec, meta = to_record(SystemOneRequest.model_validate(request))
        probs = model.probs(model.encode(tok, rec, strict=True))[0].detach().float().cpu().numpy()
        keys = meta[0]["keys"]
        if list(keys) != perm:
            raise RuntimeError(f"option order changed: {keys} != {perm}")
        for name, value in zip(keys, probs):
            total[LABEL_INDEX[name]] += float(value)
        picks.append(str(keys[int(probs.argmax())]))
    return total / len(perms), picks, len(perms)


def load_partial(path: Path) -> dict[str, dict]:
    """id -> {"probs", "flipped", "passes"} so a resumed run keeps the order-flip diagnostic."""
    done: dict[str, dict] = {}
    if not path.is_file():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[str(row["id"])] = {
            "probs": [float(x) for x in row["probs"]],
            "flipped": int(row.get("flipped", 0)),
            "passes": int(row.get("passes", 1)),
        }
    return done


def compare(path: Path, frame: pd.DataFrame, picks: list[str]) -> dict:
    other = pd.read_parquet(path).set_index("id")
    mine = pd.Series(picks, index=frame["id"].tolist())
    joined = pd.DataFrame({"mine": mine, "other": other["pred"], "gold": other["gold"]}).dropna()
    a_right = joined["mine"] == joined["gold"]
    b_right = joined["other"] == joined["gold"]
    return {
        "path": str(path),
        "n": int(len(joined)),
        "agreement": float((joined["mine"] == joined["other"]).mean()) if len(joined) else 0.0,
        "both_correct": int((a_right & b_right).sum()),
        "kev_only_correct": int((a_right & ~b_right).sum()),
        "other_only_correct": int((~a_right & b_right).sum()),
        "both_wrong": int((~a_right & ~b_right).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "hf_cache" / "kev-9b", help="Adapter dir (or Hub id).")
    parser.add_argument("--base", default=None, help="Override the base path/id (local dir for offline runs).")
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: cuda when available).")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default=None, help="Default: bf16 on cuda, fp32 on cpu.")
    parser.add_argument("--unmerged", action="store_true", help="Keep the LoRA as a PEFT module (base stays reusable).")
    parser.add_argument("--temperature", type=float, default=None, help="Override the checkpoint temperature (1.0 = raw).")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "name-that-llm" / "train.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--shuffle-options", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--compare-preds", type=Path, default=None)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = args.dtype or ("bf16" if device.startswith("cuda") else "fp32")
    args.model = Path(args.model)

    train = pd.read_csv(args.data, dtype=str)
    train["text"] = train["text"].fillna("")
    if args.limit:
        train = train.iloc[: args.limit].copy().reset_index(drop=True)

    print(f"n={len(train)} models=18  device={device} dtype={dtype_name}", flush=True)
    tok, model = load_kev(args.model, args.base, device, dtype_name, not args.unmerged, args.temperature)

    partial = args.out_dir / "kev_zeroshot.partial.jsonl"
    partial.parent.mkdir(parents=True, exist_ok=True)
    done = load_partial(partial)
    probs = np.zeros((len(train), len(LABELS)), dtype=np.float64)
    picks: list[str] = []
    flips = rows = errors = 0
    rng = random.Random(0)
    t0 = time.perf_counter()
    with partial.open("a", encoding="utf-8") as sink:
        for i, (row_id, text) in enumerate(zip(train["id"].tolist(), train["text"].tolist())):
            if row_id in done:
                cached = done[row_id]
                probs[i] = cached["probs"]
                picks.append(LABELS[int(np.argmax(cached["probs"]))])
                if cached["passes"] > 1:
                    flips += cached["flipped"]
                    rows += 1
                continue
            try:
                p, perm_picks, passes = score_one(tok, model, text, LABEL_CRITERIA, args.shuffle_options, rng)
            except Exception as exc:  # one bad row must not kill a 2,699-row run
                errors += 1
                print(f"  skip {row_id}: {type(exc).__name__}: {exc}", flush=True)
                picks.append("")
                continue
            probs[i] = p
            picks.append(LABELS[int(np.argmax(p))])
            if passes > 1:
                flips += int(len(set(perm_picks)) > 1)
                rows += 1
            sink.write(
                json.dumps(
                    {
                        "id": row_id,
                        "probs": [round(float(x), 8) for x in p],
                        "flipped": int(len(set(perm_picks)) > 1),
                        "passes": passes,
                    }
                )
                + "\n"
            )
            if (i + 1) % 250 == 0 or i + 1 == len(train) or i == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i + 1}/{len(train)}  {elapsed:.0f}s  {elapsed / (i + 1):.3f}s/row", flush=True)

    keep = [i for i, p in enumerate(picks) if p]
    y = [train["model"].iloc[i] for i in keep]
    kept = [picks[i] for i in keep]
    recalls = recall_score(y, kept, labels=LABELS, average=None, zero_division=0)
    report = {
        "model": str(args.model),
        "base": args.base,
        "n": len(keep),
        "errors": errors,
        "balanced_accuracy": float(balanced_accuracy_score(y, kept)) if keep else 0.0,
        "tfidf_oof_reference": TFIDF_OOF,
        "temperature": float(model.head.temperature),
        "merged": not args.unmerged,
        "mean_confidence": float(probs[keep].max(axis=1).mean()) if keep else 0.0,
        "option_order_flip_rate": flips / rows if rows else 0.0,
        "shuffle_options": args.shuffle_options,
        "seconds": round(time.perf_counter() - t0, 2),
        "per_class_recall": {name: float(r) for name, r in zip(LABELS, recalls)},
        "notes": [
            "zero-shot on labeled train (no training, so no GroupKFold leak)",
            "the checkpoint temperature is refit on our OOF before any blur; 2.19 is not this workload",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "kev_zeroshot.json").write_text(json.dumps(report, indent=2) + "\n")
    out = train[["id", "prompt_id", "model"]].copy().rename(columns={"model": "gold"})
    out["pred"] = picks
    out["confidence"] = probs.max(axis=1)
    out["correct"] = out["gold"] == out["pred"]
    out["prob_gold"] = [probs[i, LABEL_INDEX[g]] for i, g in enumerate(out["gold"].tolist())]
    for i, name in enumerate(LABELS):
        out[f"p_{name}"] = probs[:, i]
    out.to_parquet(args.out_dir / "kev_zeroshot_preds.parquet", index=False)

    print(json.dumps(report, indent=2), flush=True)
    print("per-class recall (worst first):", flush=True)
    for name, rec in sorted(report["per_class_recall"].items(), key=lambda kv: kv[1]):
        print(f"  {name:24s} {rec:.3f}", flush=True)
    if args.compare_preds:
        print(json.dumps(compare(args.compare_preds, train, picks), indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
