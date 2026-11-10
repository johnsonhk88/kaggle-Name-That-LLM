#!/usr/bin/env python3
"""Zero-shot Drex v1.5 on train.csv. Off-Kaggle study only — not the submit path.

Prefill + pointer head. No generation. Exact 18 LABELS.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

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
LABEL_SET = set(LABELS)
INSTRUCTIONS = "Which of these 18 language models wrote this passage? Pick exactly one author."
CRITERIA = {name: f"Written by {name}." for name in LABELS}


def load_inference(model_dir: Path):
    path = model_dir / "inference.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. Download with: hf download nace-ai/drex-v1.5 --local-dir {model_dir}"
        )
    spec = importlib.util.spec_from_file_location("drex_v15_inference", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_drex(inf, model_dir: Path):
    """Load v1.5. Hybrid Qwen3.5 cannot use state_bidir (checkpoint default is True)."""
    try:
        return inf.load(str(model_dir))
    except ValueError as exc:
        if "state_bidir" not in str(exc):
            raise
    import torch
    from kev.checkpoint import read_meta
    from kev.model import DecisionModel, load_tokenizer

    meta = read_meta(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = load_tokenizer(str(model_dir))
    model = DecisionModel(
        str(model_dir),
        tok,
        device,
        lora=None,
        head_dim=int(meta.get("head_dim", 256)),
        option_isolation=bool(meta.get("option_isolation", False)),
        dtype=dtype,
        state_bidir=False,
    )
    model.head.load_state_dict(meta["head"])
    model.head.temperature = float(meta.get("temperature", 1.0))
    model.eval()
    print("loaded with state_bidir=False (hybrid Qwen3.5)", flush=True)
    return tok, model


def request_for(text: str) -> dict:
    return {
        "state": text,
        "questions": {
            "model": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": CRITERIA,
            }
        },
    }


def parse_answer(result: dict) -> tuple[str, float, dict[str, float]]:
    block = result["answers"]["model"]
    probs = {str(k): float(v) for k, v in block["probabilities"].items()}
    choice = str(block["choice"])
    conf = float(block.get("confidence", probs.get(choice, 0.0)))
    return choice, conf, probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "hf_cache" / "drex-v1.5")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "name-that-llm" / "train.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N rows (0 = all).")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train = pd.read_csv(args.data, dtype=str)
    train["text"] = train["text"].fillna("")
    if args.limit:
        train = train.iloc[: args.limit].copy()

    inf = load_inference(args.model)
    print(f"loading {args.model} …", flush=True)
    t_load = time.perf_counter()
    tok, model = load_drex(inf, args.model)
    print(f"loaded in {time.perf_counter() - t_load:.1f}s  n={len(train)}", flush=True)

    requests = [request_for(t) for t in train["text"].tolist()]
    preds, confs, prob_rows = [], [], []
    t0 = time.perf_counter()
    bs = max(1, args.batch_size)
    for start in range(0, len(requests), bs):
        chunk = requests[start : start + bs]
        results = inf.decide_batch(tok, model, chunk)
        for result in results:
            choice, conf, probs = parse_answer(result)
            preds.append(choice)
            confs.append(conf)
            prob_rows.append(probs)
        done = start + len(chunk)
        gold = train["model"].iloc[:done]
        ba = balanced_accuracy_score(gold, preds)
        elapsed = time.perf_counter() - t0
        print(
            f"{done}/{len(requests)}  running_ba={ba:.4f}  "
            f"{elapsed:.0f}s  {elapsed / done:.3f}s/row",
            flush=True,
        )

    unknown = sorted(set(preds) - LABEL_SET)
    if unknown:
        print(f"WARNING non-LABELS predictions: {unknown}", flush=True)
    y = train["model"].tolist()
    ba = float(balanced_accuracy_score(y, preds))
    recalls = recall_score(y, preds, labels=LABELS, average=None, zero_division=0)

    report = {
        "model": str(args.model),
        "n": int(len(train)),
        "balanced_accuracy": ba,
        "tfidf_oof_reference": 0.3152249565001243,
        "seconds": round(time.perf_counter() - t0, 2),
        "unknown_labels": unknown,
        "per_class_recall": {n: float(r) for n, r in zip(LABELS, recalls)},
        "notes": [
            "zero-shot on labeled train (no training, so no GroupKFold leak)",
            "do not wire into the submit notebook unless this beats TF-IDF and is complementary",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "drex_zeroshot.json").write_text(json.dumps(report, indent=2))
    out = train[["id", "prompt_id", "model"]].copy()
    out = out.rename(columns={"model": "gold"})
    out["pred"] = preds
    out["confidence"] = confs
    out["correct"] = out["gold"] == out["pred"]
    out["prob_gold"] = [
        float(prob_rows[i].get(g, 0.0)) for i, g in enumerate(out["gold"].tolist())
    ]
    out.to_parquet(args.out_dir / "drex_zeroshot_preds.parquet", index=False)
    print(json.dumps(report, indent=2))
    print("per-class recall (worst first):")
    for name, rec in sorted(zip(LABELS, recalls), key=lambda kv: kv[1]):
        print(f"  {name:24s} {rec:.3f}")


if __name__ == "__main__":
    main()
