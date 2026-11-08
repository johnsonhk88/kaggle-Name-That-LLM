# Name That LLM

Identify which of **18 frontier LLMs** wrote a short passage.

This is a Kaggle Featured **code competition**: you upload **one notebook**, Kaggle re-runs it internet-off on ~4,000 hidden passages, and scores **balanced accuracy**.

Competition: https://www.kaggle.com/competitions/name-that-llm

| | |
|---|---|
| Submit file | **`src/name-that-llm-submit.ipynb`** (the only scored file) |
| Kaggle settings | GPU **T4 x2**, Internet **off** |
| Output | `submission.parquet` with columns `id`, `model` |
| Metric | Balanced accuracy (mean per-class recall) |
| Current backend | Word+char TF-IDF + LogisticRegression |
| Later | Fine-tune Qwen3.5-9B-Base and Kev-9B off-Kaggle, then infer in the same notebook |
| Deadlines | Join/merge **6 Jan 2027 23:59 UTC**; submit **13 Jan 2027 23:59 UTC** |

Interactive `test.csv` has **3 rows**. The scored rerun has **~4000**. Do not treat a 10-second local/interactive run as proof that the full submit will finish.

---

## 1. What is in this repo

```
dataset/name-that-llm/          train.csv, test.csv, sample_submission.parquet
src/name-that-llm-submit.ipynb  ← upload this to Kaggle
src/sample/                     public reference notebooks (not our submit)
docs/kaggle-model-import.md     how to attach Qwen / Kev on Kaggle
docs/phase-0-1.md               current status checklist
```

Do not split the submit path into extra Python modules. All scored code stays in the one notebook.

---

## 2. Run it locally (optional, before Kaggle)

Use this to check GroupKFold score and that `submission.parquet` is valid. It does **not** replace a Kaggle submit.

```bash
cd /path/to/kaggle-Name-That-LLM

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter

jupyter notebook src/name-that-llm-submit.ipynb
```

Or run all cells from the repo root without Jupyter:

```bash
source .venv/bin/activate
python - << 'PY'
import json
from pathlib import Path
nb = json.loads(Path("src/name-that-llm-submit.ipynb").read_text())
ns = {"__name__": "__main__"}
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        exec("".join(cell["source"]), ns, ns)
PY
```

You should see:

- `train=(2699, 4) test=(3, 2) models=18 prompts=150`
- 5 GroupKFold scores, then `OOF balanced_accuracy≈0.315`
- `outputs/submission.parquet` (3 rows, columns `id` and `model`)
- `outputs/tfidf_cv.json` and `outputs/t4_stub_status.json`

If `qwen_base` / `kev_adapter` are `null`, that is expected until those models are attached on Kaggle.

---

## 3. First Kaggle submission (do this now)

Get a scored notebook on the board before touching 9B models.

1. Open https://www.kaggle.com/competitions/name-that-llm and click **Join Competition**. Accept the rules (required before 6 Jan 2027).
2. **Code** → **New Notebook**.
3. File → **Upload notebook** → choose `src/name-that-llm-submit.ipynb`.
4. In notebook settings:
   - Accelerator: **GPU T4 x2** (CPU also works for this TF-IDF stage)
   - Internet: **Off**
   - Persistence / extra inputs: none required yet
5. Confirm competition data is attached. The notebook looks for:
   - `/kaggle/input/competitions/name-that-llm/train.csv`
   - `/kaggle/input/name-that-llm/train.csv`
6. **Save Version** → **Save & Run All**.
7. When it finishes, open the version → **Output** → confirm `submission.parquet`.
8. Click **Submit** on that version.

Interactive run uses 3 test rows (`light: true`) and still runs GroupKFold on all 2,699 train rows (a few minutes). The **Submit** rerun skips CV, fits on all train, and predicts the hidden ~4,000 rows.

Pick this version as a Final Submission later if you need a safe fallback.

---

## 4. Attach the two base models (for later 9B work)

Submit notebooks cannot download from Hugging Face. Import bases **now** so they are ready when fine-tunes exist.

| Role | Hugging Face | Attach as |
|---|---|---|
| Backbone | [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | Kaggle **Model** (public) |
| Decision adapter | [jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b) | Kaggle **Model** or **Dataset** (Dataset is more reliable for `head.pt`) |

Use **Base**, not Instruct, not the vision 9B.

**Import Qwen**

1. https://www.kaggle.com/models → **New Model**
2. Link `Qwen/Qwen3.5-9B-Base`, keep it public
3. Wait until `config.json` and safetensors finish syncing
4. Competition page → **Models** → Add Model → that variation

**Import Kev** (UI, or Dataset if the adapter files are dropped):

```bash
huggingface-cli download jaredpalmer/kev-9b --local-dir ./hf_cache/kev-9b
kaggle datasets init -p ./hf_cache/kev-9b
# edit dataset-metadata.json: public title such as kev-9b
kaggle datasets create -p ./hf_cache/kev-9b
```

**Attach to the same notebook**

1. Open `name-that-llm-submit` on Kaggle
2. **Add Input** → Models / Datasets → Qwen3.5-9B-Base + Kev
3. Save & Run All (internet still **off**)
4. Check printed `t4_stub_status.json`:
   - `qwen_base` / `kev_adapter` should be paths under `/kaggle/input/...`, not `null`
   - predictions stay TF-IDF until the 9B heads are implemented

Full checklist: [`docs/kaggle-model-import.md`](docs/kaggle-model-import.md).

---

## 5. How the submit notebook behaves

| Situation | What runs |
|---|---|
| Local or Kaggle **interactive** (`test.csv` has 3 rows) | GroupKFold CV + TF-IDF fit + 3-row parquet. 9B skipped (`light: true`). |
| Kaggle **Submit** (`KAGGLE_IS_COMPETITION_RERUN` set, ~4000 rows) | No CV. TF-IDF fit on all train. 9B slots are called only if those inputs exist **and** the functions are implemented. Today they still fall back to TF-IDF. |

Rules baked into the notebook:

- Labels must be exactly one of the 18 official names (`gpt-5.5`, never `gpt5.5`)
- No Hugging Face / pip downloads
- No token generation / thinking traces
- Qwen/Kev inference, when added, must be prefill + class/pointer head, `max_length=512`, sequential on T4 x2

---

## 6. After this (not in the current notebook)

Train **off Kaggle** (local RTX Pro 4500 32GB, Colab A100, or RunPod):

1. LoRA classifier on `Qwen/Qwen3.5-9B-Base`
2. Continue-train `jaredpalmer/kev-9b` as an 18-way choice model
3. Upload adapters as **public Kaggle Datasets**
4. Fill in `infer_qwen_classifier` and `infer_kev_decision` in the **same** submit notebook
5. Blend with the TF-IDF backend; keep T4 x2, internet off, ≤ 9 hours

Kev fused CUDA kernels are not T4-safe. Use a Turing DeltaNet fallback (see the GSQ path in `src/sample/0-680-modernbert-lora-sparse-ensemble.ipynb`).

---

## 7. Common failures

| Problem | Fix |
|---|---|
| Submit rejected | File must be named `submission.parquet` (not CSV). Columns exactly `id`, `model`. |
| Invalid label | Use only the 18 names printed in the notebook `LABELS` list. |
| Notebook times out on Submit | Interactive test is 3 rows. Time a ~4000-row path before adding 9B. |
| `FileNotFoundError` for train.csv | Start the notebook from the competition so `name-that-llm` is an input. |
| Internet error on Submit | Settings → Internet **Off**. Attach models/datasets; do not `from_pretrained("Qwen/...")` from the hub. |
| OOM on T4 | Do not load both 9B models at once. Load Qwen → logits → delete → load Kev. |

---

## 8. Reference notebooks

`src/sample/` is other people’s public code (starter TF-IDF, Qwen embedding ensemble, ~0.680 sparse ensemble). Copy ideas from them; do not submit those files as our entry.
