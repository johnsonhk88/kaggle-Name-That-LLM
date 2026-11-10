# Drex decision model (study only)

Drex is Nace.AI’s System One decision-model family. Same job as Kev: prefill a
state + typed questions, score options with a pointer (or readout) head, **no
generation**. It is **not** on the submit path until a local probe beats TF-IDF
and is complementary to Qwen / Kev.

Primary sources:

- Paper: [Towards Foundation Decision Models: Drex](https://arxiv.org/abs/2611.07854)
- Code: [nace-ai/drex-decision-models](https://github.com/nace-ai/drex-decision-models)
- 9B weights: [`nace-ai/drex-v1.5`](https://huggingface.co/nace-ai/drex-v1.5)
- Kev (same API, Apache-2.0 adapter): [`jaredpalmer/kev-9b`](https://huggingface.co/jaredpalmer/kev-9b)

## Family vs this contest

| Model | Size on disk | Backbone | Head | License | T4 submit |
|---|---|---|---|---|---|
| **Drex v1.5** | ~18 GB bf16 | MiMo-V2.6-Distill-Qwen-9B (from Qwen3.5-9B) | pointer + `head.pt` | Open RAIL-M | maybe, later |
| Drex DLM | 8B | Efficient-DLM-8B | pointer | **CC BY-NC** | no |
| Drex v2.0 | ~70 GB bf16 | Qwen3.6-35B-A3B (3B active) | pointer | Open RAIL-M | no |
| Drex Max | ~54 GB bf16 | Qwen3.8-27B, bidirectional full-attn | 255-way readout | Open RAIL-M | no |

Only **v1.5** is in the Qwen3.5-9B / Kev-9B size class. DLM is non-commercial.
v2.0 and Max do not fit T4 x2.

## vs Qwen-Base and Kev-9B

| | Qwen3.5-9B-Base | Kev-9B | Drex v1.5 |
|---|---|---|---|
| Job here | 18-way linear head | pointer over `LABELS` | pointer over `LABELS` |
| Weights | frozen base + our LoRA | LoRA r16 + `head.pt` on frozen Qwen-Base | **full 9B** + `head.pt` |
| License | Apache-2.0 | Apache-2.0 | **Nace.AI Open RAIL-M** |
| Shares Qwen Kaggle input? | yes | yes | **no** (third checkpoint) |
| Decision Index (public, &lt;10B) | n/a | ~41 breadth index | **58.08** (#1) |
| Local fit (RTX Pro 4500 32 GB) | yes | yes | yes (~18 GB bf16) |
| T4 x2 (2×16 GB) | shard | adapter on sharded base | shard or 8-bit |

This task is **fixed 18-way authorship**. The pointer head’s open option-set
trick is unused. Drex’s value is pretrained document/style decision skill, not
the API.

## How a passage is scored

System One `choice` (keys must be the official `LABELS` strings):

```json
{
  "state": "<passage>",
  "questions": {
    "model": {
      "type": "choice",
      "instructions": "Which of these 18 language models wrote this passage?",
      "criteria": {
        "gpt-5.5": "Written by gpt-5.5.",
        "qwen3.8-max": "Written by qwen3.8-max."
      }
    }
  }
}
```

v1.5 is a hybrid Qwen3.5 (Gated DeltaNet + full attention). Hybrids cannot use
the packed block-causal mask, so each question is a causal row: cached state +
that question. One question per passage ⇒ one prefill per row. Nothing is
decoded. `kev.serve` fused CUDA graphs stay **off** (not T4-safe; not needed
locally either).

## License (why we do not distill)

Open RAIL-M allows personal / research use. Redistributing weights as a Kaggle
Dataset is Distribution: keep `LICENSE`, keep RAIL-M on derivatives.

**Do not** train Qwen or Kev on Drex probabilities. The license treats models
trained on Drex output as Derivatives. Track A below copies the *recipe* only.

Attachment A also restricts competing decision products above a $1M
revenue/funding line. A Kaggle authorship notebook is research; do not ship
Drex as a hosted System One clone.

## Two tracks

### Track A (default) — steal the recipe, keep Apache-2.0

Apply to continue-train `jaredpalmer/kev-9b` off-Kaggle:

1. Pack `state=passage`, one `choice` over the 18 labels.
2. Shuffle option order every epoch (pointer models are order-sensitive).
3. SFT: cross-entropy on the gold author. Optional permutation-consistency KL.
4. Fit one temperature on GroupKFold OOF (Kev ships T=2.19; refit here).
5. Soft labels, if any, come from TF-IDF or our Qwen LoRA — **not** Drex.

Qwen stays a plain 18-way linear head, `max_length=512`, language-only Base.

### Track B — run v1.5 as a third expert

Only if the zero-shot / continue-train probe beats TF-IDF OOF (~0.315) **and**
errors are complementary to Qwen/Kev.

Then: public Dataset with weights + `head.pt` + LICENSE, sequential load after
Qwen then Kev, Turing DeltaNet fallback, blend logits. Do not add this slot to
the submit notebook while Qwen/Kev are still stubs.

## Local probe

GPU: RTX Pro 4500 32 GB (this machine). Separate env: `.venv-9b` (not the
TF-IDF `.venv`). Weights: `hf_cache/drex-v1.5/` (gitignored).

```bash
python3 -m venv .venv-9b
source .venv-9b/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r tools/requirements-9b.txt
hf download nace-ai/drex-v1.5 --local-dir hf_cache/drex-v1.5
python tools/drex_zeroshot_probe.py --limit 32   # smoke
python tools/drex_zeroshot_probe.py              # all 2,699 train rows
```

Writes `outputs/drex_zeroshot.json` and `outputs/drex_zeroshot_preds.parquet`.
Zero-shot uses the full labeled train set (no training, so no GroupKFold leak).
Continue-train later must GroupKFold on `prompt_id`.

## Do not

- Put Drex in `src/name-that-llm-submit.ipynb` yet
- Import Drex Max / v2.0 / DLM on Kaggle
- `from_pretrained("nace-ai/...")` inside the submit notebook
- Distill Drex into Kev or Qwen
- Load Drex and Qwen in GPU memory at the same time
