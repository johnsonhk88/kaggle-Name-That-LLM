# Phase 0–1 status

Submit code is **one notebook**: `src/name-that-llm-submit.ipynb`.

## In the notebook

- [x] Exact 18 labels
- [x] Local + Kaggle data paths
- [x] GroupKFold TF-IDF baseline (skipped on hidden rerun)
- [x] `submission.parquet` asserts
- [x] T4 stub: discover Qwen/Kev inputs, light-mode skip, TF-IDF fallback

## You do on Kaggle

- [ ] Join the competition
- [ ] Upload `src/name-that-llm-submit.ipynb` (T4 x2, internet off) and submit
- [ ] Import `Qwen/Qwen3.5-9B-Base` and `jaredpalmer/kev-9b` (see `docs/kaggle-model-import.md`)
- [ ] Re-run the same notebook with those inputs attached; check `t4_stub_status.json` paths

## Not in this phase

Qwen LoRA training, Kev continue-train, T4 DeltaNet kernels, extra data.
