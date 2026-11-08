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
- Kev fused CUDA kernels are not T4-safe.

## Reference only

`src/sample/` is public Kaggle notebooks, not our submit path.
`docs/kaggle-model-import.md` is the base-model attach checklist.
