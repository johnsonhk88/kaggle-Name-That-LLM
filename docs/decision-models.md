# Decision models for this contest

A decision model reads a **state** (the passage) plus **typed questions** and returns a
probability for every option, in one prefill, without generating text. That shape is a good
match for a fixed 18-way task — better than a reranker, which needs one forward pass per
candidate label.

Two families are on the shortlist: **Kev** (adapter on our own base) and **Decision 3.0**
(`vllm-sr/d3-*`). Drex is kept as prior art only.

| Model | Params | Backbone | Jev 0.3.2: **score** / public / same-skill / new-domain | Kaggle T4 |
|---|---|---|---|---|
| `vllm-sr/d3-flash` | 8.39B | Qwen3.5-9B | **55.8** / 57.8 / 54.6 / 48.6 | **no** (16.8 GB BF16) |
| `vllm-sr/d3-mini` | 4.54B | **Qwen3.5-4B** | **52.6** / 54.9 / 51.9 / 45.6 | **yes** (9.1 GB) |
| `nace-ai/drex-v1.5` | ~9B | MiMo-V2.6-Distill-Qwen-9B | 49.7 / 58.1 / 49.7 / 39.7 | maybe |
| `jaredpalmer/kev-9b` | 45.4M adapter | Qwen3.5-9B | not on this board; breadth-v1 index **41.0** on its card | shares base |

`d3-nano` (2B), `d3-lite` (0.9B) and `d3-edge` (0.6B) exist and are cheap, but they are below the
52.6 floor and will not separate `gpt-5.6-sol` from `gpt-5.6-terra`. Do not add them.

## Why d3-mini on Kaggle, d3-flash off it

- The released d3 runtime **refuses multi-device**: `device_map="auto"` collapses to the default
  device and a dict with more than one entry raises. d3-flash's 16.8 GB BF16 does not fit one
  16 GB T4; d3-mini's 9.1 GB does.
- d3-mini is also the **only shortlist member not built on the 9B base**. Qwen LoRA, Kev and
  d3-flash are three views of one prior; d3-mini is a different backbone, which is worth more
  for confusable siblings than its 3.2 index gap costs.
- d3-flash stays off-Kaggle as a **teacher**: Apache-2.0 permits distilling its probabilities
  into d3-mini or into the Qwen head. Drex's Open RAIL-M did not.

## How a passage is scored

d3 (`AutoModel.from_pretrained(..., trust_remote_code=True)`):

```python
res = model.system_one(
    state=passage,
    questions={"model": {
        "type": "choice",
        "instructions": "Which of these 18 language models wrote this passage?",
        "criteria": {name: f"Written by {name}." for name in LABELS},
    }},
)
# res["answers"]["model"] -> {"type", "choice", "probabilities": {label: p}, "confidence"}
```

Kev (PEFT adapter + pointer head, one base load):

```python
from kev.checkpoint import Checkpoint, LoadOptions
from kev.api import SystemOneRequest, to_record

ckpt = Checkpoint("hf_cache/kev-9b")
ckpt.meta.base = "hf_cache/qwen3.5-9b-base"   # offline / Kaggle: drop the Hub revision pin
ckpt.meta.base_revision = None
tok, model = ckpt.load("cuda", LoadOptions(dtype=torch.bfloat16, merge=False))
rec, meta = to_record(SystemOneRequest.model_validate({
    "state": passage,
    "questions": {"model": {"type": "choice", "instructions": INSTRUCTIONS,
                            "criteria": LABEL_CRITERIA}},
}))
probs = model.probs(model.encode(tok, rec, strict=True))[0]
```

`merge=False` is the submit-path shape: the adapter stays a PEFT module, so the frozen base is
never rewritten and can host a second adapter afterwards. `merge=True` is fine for a standalone
probe and slightly faster.

Kev ships **T = 2.19**, fitted on its own held-out public sets. Refit the temperature on
GroupKFold OOF before blending; 2.19 is not this workload.

## The zero-shot risk

Both models are trained on semantic decisions (routing, NLI, intent, policy). Authorship style is
out of domain, and the label strings are opaque: an option called `gpt-5.6-luna` with description
"Written by gpt-5.6-luna." carries no discriminative signal.

Two mitigations, both in `tools/d3_zeroshot_probe.py`:

1. `--arm label` — the honest zero-shot baseline.
2. `--arm desc` — per-class distinctive character n-grams fitted **on the fold train split only**,
   e.g. `gpt-5.5: Written by gpt-5.5. Style markers: " the", ", and", "—"`. `GroupKFold` on
   `prompt_id`, so no prompt leaks across folds.

Kill criteria: `desc` must beat the TF-IDF OOF baseline (**0.315**) and its errors must be
complementary to the Qwen/Kev OOF before it earns a slot. A raw index win is not enough; the
public board's identical-code spread is ~0.012.

## Track A — continue-train if a probe is promising

Reuse the Kev recipe, do not invent a new one:

1. Pack `state=passage`, one `choice` over the 18 `LABELS`.
2. Shuffle option order every epoch (pointer models are order-sensitive; `--shuffle-options`
   measures the serving-side sensitivity).
3. Cross-entropy on the gold author; optional permutation-consistency KL.
4. Fit one temperature on GroupKFold OOF.
5. Soft labels may come from our Qwen LoRA or from `d3-flash` (Apache-2.0). **Not** from Drex.

## Probe commands

```bash
source .venv-9b/bin/activate                                  # torch cu128 + transformers 5.17
hf download vllm-sr/d3-mini --local-dir hf_cache/d3-mini      # 9.1 GB, once
python tools/d3_zeroshot_probe.py --local --model hf_cache/d3-mini --limit 32
python tools/d3_zeroshot_probe.py --local --model hf_cache/d3-mini --arm both
python tools/kev_adapter_probe.py --model hf_cache/kev-9b --base hf_cache/qwen3.5-9b-base
```

Both probes checkpoint every row (`outputs/*.partial.jsonl`), so an interrupted run resumes and does
not re-score. Delete the partial to start over.

## Do not

- Put d3 or Kev in `src/name-that-llm-submit.ipynb` before a probe beats TF-IDF.
- Attach `d3` (26B) or `d3-flash` (16.8 GB) to the Kaggle submit notebook.
- Load two 9B checkpoints in GPU memory at once; swap adapters instead.
- Distill Drex or redistribute its weights.
- Trust a single public/probe score: run the probe on 2 answers per prompt as well as all 18
  (the forum result is 0.91 vs 0.13 correlation with the board).
