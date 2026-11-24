# Kaggle model import (Phase 0)

Submit notebooks run **internet off**. Import the bases now. Publish fine-tunes later as Datasets.

## 1. Join the competition

1. Open https://www.kaggle.com/competitions/name-that-llm
2. **Join Competition** and accept the rules (required before 6 Jan 2027 23:59 UTC).
3. Confirm GPU setting **T4 x2** and **Internet off** on any submit notebook.

## 2. Import the three models (public)

| Role | Hugging Face | Kaggle form | Size | Notes |
|---|---|---|---|---|
| Backbone | [`Qwen/Qwen3.5-9B-Base`](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | **Model** | ~18 GB | Language-only Base, not Instruct, not 9B-VL |
| Decision adapter | [`jaredpalmer/kev-9b`](https://huggingface.co/jaredpalmer/kev-9b) | **Dataset** | **0.55 GB** | PEFT LoRA + `head.pt`; **no base weights**, so a Dataset is enough |
| Third expert | [`vllm-sr/d3-mini`](https://huggingface.co/vllm-sr/d3-mini) | **Model** | 9.1 GB | Decision 3.0, Qwen3.5-4B, Apache-2.0 |

Do **not** import:

- [`vllm-sr/d3-flash`](https://huggingface.co/vllm-sr/d3-flash) — 16.8 GB BF16 does not fit a 16 GB T4, and the d3 runtime refuses to shard across two. Off-Kaggle teacher only.
- [`nace-ai/drex-v1.5`](https://huggingface.co/nace-ai/drex-v1.5) — Open RAIL-M, not on the shortlist. See `docs/decision-models.md`.
- `d3` / `d3-turbo` / `d3-ultra` — 26B.

### Model import (UI)

1. https://www.kaggle.com/models → **New Model**
2. Link the Hugging Face repo, keep it **public**, Apache-2.0.
3. Wait until files finish syncing. For d3-mini that means `decision_config.json`, `readout.safetensors` and **both** `model-*.safetensors` shards.

### Adapter fallback (recommended for Kev)

If a Model import drops `head.pt` or the adapter files:

```bash
huggingface-cli download jaredpalmer/kev-9b --local-dir ./hf_cache/kev-9b
kaggle datasets init -p ./hf_cache/kev-9b
# edit dataset-metadata.json: title "kev-9b", public
kaggle datasets create -p ./hf_cache/kev-9b
```

Same pattern later for **your** LoRA + `head.pt`.

### d3 runtime requirements

d3-mini needs `transformers==5.17.0` and `trust_remote_code`. If the Kaggle image is older, vendor
the wheels as a Dataset (the public notebooks already do this for Qwen3.5; see
`src/sample/0-680-modernbert-lora-sparse-ensemble.ipynb` and its `qwen38-offline-linux` input).
`flash-linear-attention` is optional and **not T4-safe** — do not enable fused kernels.

## 3. Attach on this competition

1. Competition page → **Models** → Add Model → the Qwen3.5-9B-Base variation.
2. Open a notebook → **Add Input** → Models / Datasets → Qwen base + Kev + d3-mini.
3. Paths look like:

```
/kaggle/input/models/<owner>/<slug>/transformers/<variation>/<version>
/kaggle/input/<your-kev-dataset>/
```

The submit notebook searches `/kaggle/input` for `config.json` / `adapter_config.json` /
`head.pt` / `decision_config.json`.

## 4. Also attach

- Competition data `name-that-llm` (automatic if you start from the competition).
- Optional: a `transformers` wheel Dataset **if** the default image cannot `import transformers.models.qwen3_5` or is older than 5.17.

## 5. Smoke test (interactive, internet off)

Use `src/name-that-llm-submit.ipynb`:

- [ ] Notebook has Internet **off**, GPU **T4 x2**
- [ ] `train.csv` / `test.csv` resolve
- [ ] Status JSON shows discovered Qwen / Kev / d3 paths (or explicit `null` if not attached yet)
- [ ] `submission.parquet` has columns `id`, `model` and only the 18 official labels
- [ ] Interactive run stays on TF-IDF (`light: true`) because `test.csv` has 3 rows
- [ ] After a real **Submit**, `KAGGLE_IS_COMPETITION_RERUN` is set; then the expert slots run (still TF-IDF fallback until Phase 2–3)

## 6. Fine-tunes (later, Phase 5)

Train off-Kaggle. Upload **public** Datasets:

- `ntl-qwen35-9b-lora` — adapter + 18-way `head.pt` + `label_map.json`
- `ntl-kev-9b-ntl` — continued Kev adapter + calibrated `head.pt`
- `ntl-d3-mini-ntl` — only if the zero-shot probe clears TF-IDF

Do not merge a full 9B checkpoint unless quantization is solved. Adapters are enough.

## 7. T4 rules of the road

- **One 9B load**: Qwen base once → our LoRA head → `set_adapter(kev)` + `head.pt`. Never delete and reload.
- Use `LoadOptions(merge=False)` for Kev in the submit path; `merge=True` destroys the adapter module.
- `max_length=512`, language-only, **no generation**, no thinking.
- d3 runs on **one** device (`device="cuda:0"`). Only `d3-mini` fits a T4.
- Shard one 9B across both T4s in bf16 before trying bitsandbytes (DeltaNet).
- Kev fused CUDA graphs and d3 fused linear-attention are not T4-safe; use the Turing DeltaNet fallback from `src/sample/0-680-modernbert-lora-sparse-ensemble.ipynb`.
