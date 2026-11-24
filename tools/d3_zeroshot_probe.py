#!/usr/bin/env python3
"""Zero-shot Decision 3.0 (d3) on train.csv. Off-Kaggle study only — not the submit path.

One `choice` question over the 18 LABELS per passage: one forward pass, probabilities only,
no generation. Two criteria arms:

  label   criteria = {"gpt-5.5": "Written by gpt-5.5.", ...}          (no training, no leak)
  desc    per-fold style descriptions fitted on the GroupKFold train  (leak-free by fold)

The `desc` arm exists because the labels are opaque strings: an option called "gpt-5.6-luna"
gives a decision model nothing to discriminate on.

Usage
  python tools/d3_zeroshot_probe.py --limit 32                     # smoke
  python tools/d3_zeroshot_probe.py                                # Hub d3-mini, label arm
  python tools/d3_zeroshot_probe.py --arm both                     # + per-fold desc arm
  python tools/d3_zeroshot_probe.py --local --model hf_cache/d3-mini
  python tools/d3_zeroshot_probe.py --shuffle-options 4            # order-robustness check
  python tools/d3_zeroshot_probe.py --compare-preds outputs/kev_zeroshot_label_preds.parquet

Writes outputs/d3_zeroshot_<arm>.json and outputs/d3_zeroshot_<arm>_preds.parquet.
d3 runs on ONE device; d3-flash (16.8 GB BF16) does not fit a 16 GB T4, d3-mini (9.1 GB) does.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold

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
TFIDF_OOF = 0.3152249565001243


def load_model(model_id: str, device: str | None, local: bool):
    from transformers import AutoModel

    kwargs: dict = {"trust_remote_code": True, "local_files_only": local}
    if device:
        kwargs["device"] = device
    model = AutoModel.from_pretrained(model_id, **kwargs)
    print(f"loaded {model_id} (device={device or 'runtime default'})", flush=True)
    return model


def score_one(model, text: str, criteria: dict[str, str], shuffle: int, rng: random.Random):
    """-> (probabilities in LABELS order, list of picks, number of forward passes)."""
    order = list(criteria)
    perms = [order] if shuffle <= 1 else [rng.sample(order, len(order)) for _ in range(shuffle)]
    total = np.zeros(len(LABELS), dtype=np.float64)
    picks: list[str] = []
    for perm in perms:
        block = model.system_one(
            state=text,
            questions={
                "model": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": {name: criteria[name] for name in perm},
                }
            },
        )["answers"]["model"]
        if "error" in block:
            raise RuntimeError(f"{block.get('error')}: {block.get('message')}")
        probs = block["probabilities"]
        for name in perm:
            total[LABEL_INDEX[name]] += float(probs.get(name, 0.0))
        picks.append(max(perm, key=lambda name: float(probs.get(name, 0.0))))
    return total / len(perms), picks, len(perms)


def style_criteria(train: pd.DataFrame, topk: int, analyzer: str) -> dict[str, str]:
    """Per-class distinctive n-grams, fitted on one training fold only."""
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(2, 5) if analyzer == "char_wb" else (1, 2),
        min_df=5,
        max_features=200_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vec.fit_transform(train["text"].tolist())
    names = vec.get_feature_names_out()
    glob = np.asarray(matrix.mean(axis=0)).ravel()
    out: dict[str, str] = {}
    for label in LABELS:
        rows = train["model"].to_numpy() == label
        if not rows.any():
            out[label] = f"Written by {label}."
            continue
        mean = np.asarray(matrix[rows].mean(axis=0)).ravel()
        terms: list[str] = []
        for i in np.argsort(-(mean - glob)):
            term = str(names[i]).strip()
            if len(term) >= 2 and term not in terms:
                terms.append(term)
            if len(terms) == topk:
                break
        out[label] = f"Written by {label}. Style markers: " + ", ".join(terms)
    return out


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


def score_frame(model, frame: pd.DataFrame, criteria, tag: str, args, rng) -> tuple[np.ndarray, list[str], int, float]:
    """Score every row; resume from outputs/<tag>.partial.jsonl. -> (probs, picks, errors, flip_rate)."""
    partial = args.out_dir / f"{tag}.partial.jsonl"
    partial.parent.mkdir(parents=True, exist_ok=True)
    done = load_partial(partial)
    probs = np.zeros((len(frame), len(LABELS)), dtype=np.float64)
    picks: list[str] = []
    flips = rows = errors = 0
    t0 = time.perf_counter()
    with partial.open("a", encoding="utf-8") as sink:
        for i, (row_id, text) in enumerate(zip(frame["id"].tolist(), frame["text"].tolist())):
            if row_id in done:
                cached = done[row_id]
                probs[i] = cached["probs"]
                picks.append(LABELS[int(np.argmax(cached["probs"]))])
                if cached["passes"] > 1:
                    flips += cached["flipped"]
                    rows += 1
                continue
            try:
                p, perm_picks, passes = score_one(model, text, criteria, args.shuffle_options, rng)
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
            if (i + 1) in (1, 32) or (i + 1) % 250 == 0 or i + 1 == len(frame):
                elapsed = time.perf_counter() - t0
                print(f"  {tag} {i + 1}/{len(frame)}  {elapsed:.0f}s  {elapsed / (i + 1):.3f}s/row", flush=True)
    flip_rate = flips / rows if rows else 0.0
    return probs, picks, errors, flip_rate


def report_for(y: list[str], picks: list[str], probs: np.ndarray, errors: int, flip_rate: float, args, arm: str) -> dict:
    keep = [i for i, p in enumerate(picks) if p]
    y_kept = [y[i] for i in keep]
    p_kept = [picks[i] for i in keep]
    recalls = recall_score(y_kept, p_kept, labels=LABELS, average=None, zero_division=0)
    conf = probs[keep].max(axis=1) if keep else np.zeros(1)
    return {
        "model": str(args.model),
        "arm": arm,
        "n": len(keep),
        "errors": errors,
        "balanced_accuracy": float(balanced_accuracy_score(y_kept, p_kept)) if keep else 0.0,
        "tfidf_oof_reference": TFIDF_OOF,
        "mean_confidence": float(conf.mean()),
        "option_order_flip_rate": flip_rate,
        "shuffle_options": args.shuffle_options,
        "seconds": None,
        "per_class_recall": {name: float(r) for name, r in zip(LABELS, recalls)},
        "notes": [
            "zero-shot on labeled train; the desc arm fits its criteria on the fold train split only",
            "do not wire into the submit notebook unless this beats TF-IDF and is complementary",
        ],
    }


def write_outputs(out_dir: Path, arm: str, frame: pd.DataFrame, picks: list[str], probs: np.ndarray, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"d3_zeroshot_{arm}.json").write_text(json.dumps(report, indent=2) + "\n")
    out = frame[["id", "prompt_id", "model"]].copy().rename(columns={"model": "gold"})
    out["pred"] = picks
    out["confidence"] = probs.max(axis=1)
    out["correct"] = out["gold"] == out["pred"]
    out["prob_gold"] = [probs[i, LABEL_INDEX[g]] for i, g in enumerate(out["gold"].tolist())]
    for i, name in enumerate(LABELS):
        out[f"p_{name}"] = probs[:, i]
    out.to_parquet(out_dir / f"d3_zeroshot_{arm}_preds.parquet", index=False)


def compare(path: Path, frame: pd.DataFrame, picks: list[str]) -> dict:
    other = pd.read_parquet(path).set_index("id")
    mine = pd.Series(picks, index=frame["id"].tolist())
    joined = pd.DataFrame({"mine": mine, "other": other["pred"], "gold": other["gold"]}).dropna()
    agree = float((joined["mine"] == joined["other"]).mean()) if len(joined) else 0.0
    a_right = joined["mine"] == joined["gold"]
    b_right = joined["other"] == joined["gold"]
    return {
        "path": str(path),
        "n": int(len(joined)),
        "agreement": agree,
        "both_correct": int((a_right & b_right).sum()),
        "d3_only_correct": int((a_right & ~b_right).sum()),
        "other_only_correct": int((~a_right & b_right).sum()),
        "both_wrong": int((~a_right & ~b_right).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Hub id or local dir (default: vllm-sr/d3-mini, or hf_cache/d3-mini with --local).")
    parser.add_argument("--local", action="store_true", help="Load a local directory with local_files_only=True.")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu. d3 runs on ONE device; do not pass device_map='auto'.")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "name-that-llm" / "train.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--arm", choices=("label", "desc", "both"), default="label")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--desc-analyzer", choices=("char_wb", "word"), default="char_wb")
    parser.add_argument("--desc-topk", type=int, default=10)
    parser.add_argument("--shuffle-options", type=int, default=1, help="Average over N option orders (1 = prompt order).")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N rows (0 = all).")
    parser.add_argument("--compare-preds", type=Path, default=None, help="Another probe's *_preds.parquet for a complementarity count.")
    args = parser.parse_args()

    args.model = args.model or (str(ROOT / "hf_cache" / "d3-mini") if args.local else "vllm-sr/d3-mini")
    train = pd.read_csv(args.data, dtype=str)
    train["text"] = train["text"].fillna("")
    if args.limit:
        train = train.iloc[: args.limit].copy().reset_index(drop=True)
    y = train["model"].tolist()
    rng = random.Random(0)

    print(f"n={len(train)} models=18  arm={args.arm}", flush=True)
    model = load_model(args.model, args.device, args.local)

    arms = ("label", "desc") if args.arm == "both" else (args.arm,)
    all_picks: dict[str, list[str]] = {}
    for arm in arms:
        t0 = time.perf_counter()
        if arm == "label":
            criteria = {name: f"Written by {name}." for name in LABELS}
            probs, picks, errors, flip = score_frame(model, train, criteria, "d3_zeroshot_label", args, rng)
        else:
            probs = np.zeros((len(train), len(LABELS)), dtype=np.float64)
            picks = [""] * len(train)
            errors = 0
            flips = rows = 0
            splitter = GroupKFold(n_splits=args.folds)
            for fold, (fit_idx, val_idx) in enumerate(splitter.split(train, groups=train["prompt_id"])):
                criteria = style_criteria(train.iloc[fit_idx], args.desc_topk, args.desc_analyzer)
                part = train.iloc[val_idx].reset_index(drop=True)
                p, picks_part, err, flip_part = score_frame(
                    model, part, criteria, f"d3_zeroshot_desc_f{fold}", args, rng
                )
                probs[val_idx] = p
                for j, i in enumerate(val_idx):
                    picks[i] = picks_part[j]
                errors += err
                flips += int(flip_part * len(part))
                rows += len(part)
                print(f"  fold {fold}: n={len(part)} errors={err}", flush=True)
            flip = flips / rows if rows else 0.0
        report = report_for(y, picks, probs, errors, flip, args, arm)
        report["seconds"] = round(time.perf_counter() - t0, 2)
        write_outputs(args.out_dir, arm, train, picks, probs, report)
        all_picks[arm] = picks
        print(json.dumps({k: v for k, v in report.items() if k != "per_class_recall"}, indent=2), flush=True)
        print("per-class recall (worst first):", flush=True)
        for name, rec in sorted(report["per_class_recall"].items(), key=lambda kv: kv[1]):
            print(f"  {name:24s} {rec:.3f}", flush=True)

    if args.compare_preds and all_picks:
        first = all_picks[arms[0]]
        print(json.dumps(compare(args.compare_preds, train, first), indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
