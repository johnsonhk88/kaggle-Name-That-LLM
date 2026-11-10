# Kaggle model import (Phase 0)

Submit notebooks run **internet off**. Import bases now. Publish fine-tunes later as Datasets.

## 1. Join the competition

1. Open https://www.kaggle.com/competitions/name-that-llm
2. **Join Competition** and accept the rules (required before 6 Jan 2027 23:59 UTC).
3. Confirm GPU setting **T4 x2** and **Internet off** on any submit notebook.

## 2. Import the two bases (public)

| Role | Hugging Face | Kaggle form | Notes |
|---|---|---|---|
| Backbone | [`Qwen/Qwen3.5-9B-Base`](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | **Kaggle Model** | Language-only Base, not Instruct, not 9B-VL |
| Decision adapter | [`jaredpalmer/kev-9b`](https://huggingface.co/jaredpalmer/kev-9b) | Model **or Dataset** | PEFT adapter + `head.pt`; Dataset is more reliable |

Do **not** import [`nace-ai/drex-v1.5`](https://huggingface.co/nace-ai/drex-v1.5) yet (Open RAIL-M, full 18 GB checkpoint, not an adapter on Qwen-Base). Study notes: `docs/drex-decision-model.md`.

### Model import (UI)

1. https://www.kaggle.com/models → **New Model**
2. Link the Hugging Face repo, keep it **public**, Apache-2.0.
3. Wait until files (especially `config.json` / safetensors) finish syncing.

### Adapter fallback (recommended for Kev)

If Models import drops `head.pt` or adapter files:

```bash
huggingface-cli download jaredpalmer/kev-9b --local-dir ./hf_cache/kev-9b
kaggle datasets init -p ./hf_cache/kev-9b
# edit dataset-metadata.json: title "kev-9b", public
kaggle datasets create -p ./hf_cache/kev-9b
```

Same pattern later for **your** LoRA + `head.pt`.

## 3. Attach on this competition

1. Competition page → **Models** → Add Model → the Qwen3.5-9B-Base variation.
2. Open a notebook → **Add Input** → Models / Datasets → Qwen base + Kev.
3. Paths look like:

```
/kaggle/input/models/<owner>/<slug>/transformers/<variation>/<version>
/kaggle/input/<your-kev-dataset>/
```

The submit notebook searches `/kaggle/input` for `config.json` / `adapter_config.json` / `head.pt`.

## 4. Also attach

- Competition data `name-that-llm` (automatic if you start from the competition).
- Optional: a `transformers` wheel Dataset **if** the default image cannot `import transformers.models.qwen3_5`.

## 5. Smoke test (interactive, internet off)

Use `src/name-that-llm-submit.ipynb`:

- [ ] Notebook has Internet **off**, GPU **T4 x2**
- [ ] `train.csv` / `test.csv` resolve
- [ ] Status JSON shows discovered Qwen / Kev paths (or explicit `null` if not attached yet)
- [ ] `submission.parquet` has columns `id`, `model` and only the 18 official labels
- [ ] Interactive run stays on TF-IDF (`light: true`) because `test.csv` has 3 rows
- [ ] After a real **Submit**, `KAGGLE_IS_COMPETITION_RERUN` is set; then 9B slots run (still TF-IDF fallback until Phase 2–3)

## 6. Fine-tunes (later, Phase 5)

Train off-Kaggle. Upload **public** Datasets:

- `ntl-qwen35-9b-lora` — adapter + 18-way `head.pt` + `label_map.json`
- `ntl-kev-9b-ntl` — continued Kev adapter + calibrated `head.pt`

Do not merge a full 9B checkpoint unless quantization is solved. Adapters are enough.

## 7. T4 rules of the road

- Sequential: load Qwen → write logits → delete → `empty_cache` → load Kev.
- `max_length=512`, language-only, **no generation**, no thinking.
- Shard one model across both T4s in bf16 before trying bitsandbytes (DeltaNet).
- Kev fused CUDA graphs are not T4-safe; use the Turing DeltaNet fallback from `src/sample/0-680-modernbert-lora-sparse-ensemble.ipynb`.
