# Name That LLM

Kaggle Featured code competition: 18-way LLM authorship attribution.
Metric is balanced accuracy. Submit **one** notebook.

## Submit code

All scored code lives in `src/name-that-llm-submit.ipynb`.
Do not split the submit path into a Python package.

Kaggle settings: GPU T4 x2, Internet **off**, write `submission.parquet`.

## Conventions

- Exact 18 labels only (`gpt-5.5`, never `gpt5.5`).
- Validate with GroupKFold on `prompt_id`.
- Interactive `test.csv` is 3 rows; hidden rerun is ~4000.
- No Hugging Face downloads in the submit notebook.
- Prefill + class/pointer head only. No generation, no thinking.
- Use `Qwen/Qwen3.5-9B-Base` (language-only), not Instruct.

## Models (final shortlist — do not add others)

| Slot | Model | Kaggle form | Role |
|---|---|---|---|
| 1 | `Qwen/Qwen3.5-9B-Base` + our LoRA + 18-way linear head | Model + Dataset | primary, only model trained on this task |
| 2 | `jaredpalmer/kev-9b` — PEFT LoRA r16 + `head.pt` (0.55 GB, **no base weights**) | Dataset | decision prior; **one base load**, swap adapters |
| 3 | `vllm-sr/d3-mini` — Decision 3.0, 4.54B, Qwen3.5-4B | Model | third expert, only non-9B backbone |
| teacher | `vllm-sr/d3-flash` — Decision 3.0, 8.39B | **off-Kaggle only** | soft labels; 16.8 GB does **not** fit a 16 GB T4 |

- No Qwen3-Reranker. No Drex. `docs/decision-models.md` explains why.
- Kev and d3 are zero-shot out-of-domain here. Probe before believing the index.
- d3 runs on **one** device (the runtime refuses `device_map` across GPUs). Only `d3-mini` fits a T4.
- Kev fused CUDA kernels and d3 fused linear-attention are not T4-safe; use the Turing fallback.
- Do not merge the Kev adapter into the base in the submit path: `LoadOptions(merge=False)` keeps the base reusable.
- `max_length=512`. d3 truncates nothing — it errors with `max_length_exceeded` instead.

## Study tools

- `tools/d3_zeroshot_probe.py` — d3 `choice` over the 18 labels, label + per-fold desc arms.
- `tools/kev_adapter_probe.py` — Kev pointer head, same folds.
- `tools/drex_zeroshot_probe.py` — prior art only (Open RAIL-M, not on the shortlist).

## Reference only

`src/sample/` is public Kaggle notebooks, not our submit path.
`docs/kaggle-model-import.md` is the base-model attach checklist.
