# Turing Tag

NER-aware machine translation. Named entities stay — everything else shifts.

The problem: translation APIs don't know that "Athmanandam" is a name, not two Hindi words. They'll happily translate it to "आत्मानंदम" (soul-happy). Turing Tag runs NER first, masks the entities, translates the rest, and stitches it back together. The name survives.

## What this does

```
Input: "Taj Mahal was built by Shah Jahan in Agra"
                ↓
        NER identifies: Taj Mahal (geo), Shah Jahan (per), Agra (geo)
                ↓
        Masked: "__ENT0__ was built by __ENT1__ in __ENT2__"
                ↓
        Translated: "__ENT0__ fue construido por __ENT1__ en __ENT2__"
                ↓
Output: "Taj Mahal fue construido por Shah Jahan en Agra"
```

Entities are color-coded in the UI, collected as they appear, and preserved across any target language. The system uses a hybrid approach combining **ML Models** with a **Lexicon Gazetteer** to ensure 100% detection rate for high-priority brands and names.

## Architecture

```
pipeline/          NER model training (CRF, BiLSTM-CRF, BERT, DistilBERT, Keras BiLSTM)
api/               FastAPI backend — NER + translation service
web/               React + TypeScript + Vite frontend
monitoring/        Prometheus config
tests/             pytest suite
```

**Pipeline** trains four NER models on the GMB corpus (with a fifth Keras BiLSTM included as a baseline), tracks experiments with MLflow, and promotes the best one. **API** loads the winning model at startup and exposes REST + WebSocket endpoints. Translation is pluggable — swap Google Translate for MyMemory or MarianMT (local) via an env var. **Frontend** has request and realtime modes, highlights entities inline, and collects them in a sidebar.

## Models

Five NER models. The four PyTorch models share a single 70/15/15 train/val/test split (random_state=42); the Keras model was trained earlier on its own split and is included as a baseline reference.

| Model | Type | Test F1 | Notes |
|---|---|---|---|
| **CRF** | Classical | **0.8435** | Hand-crafted features (case, prefix/suffix, POS, ±2 context window). CPU only. |
| DistilBERT-NER | Transformer (small) | 0.8254 | Fine-tuned `distilbert-base-uncased`, 66M params, ~40% faster than BERT. |
| BERT-NER | Transformer | 0.8225 | Fine-tuned `bert-base-uncased`, 110M params. |
| Keras BiLSTM (`.h5`) | Deep learning | 0.8197 | TensorFlow/Keras. Embedding(35,178×128) → BiLSTM(64) → TimeDistributed Dense(17, softmax). No CRF layer. |
| BiLSTM-CRF | Deep learning | 0.8183 | PyTorch, GPU. Embedding(128) → BiLSTM(256) → custom CRF layer with viterbi decoding. |

**Why CRF wins on this dataset:**
- Hand-crafted features (capitalization, suffix patterns, POS tags) inject linguistic priors that the deep models have to learn from scratch on only ~33K training sentences.
- The CRF layer enforces hard tag-transition constraints (`I-per` cannot follow `B-org`); the Keras BiLSTM lacks this entirely and emits per-token softmax decisions.
- Tens of thousands of parameters vs BERT's 110M means no overfitting risk on a small corpus.
- The transformer models are within 0.02 F1 of each other — they'd likely overtake CRF with more data, longer training, or a learning-rate scheduler.

The best by F1 gets registered to the MLflow Model Registry and served by the API.

## Setup

```bash
# clone and enter
git clone https://github.com/your-org/turing_tag.git
cd turing_tag

# backend
# NOTE: It is recommended to use Python 3.12. Avoid 3.14 (experimental) due to ML library locks.
py -3.12 -m venv venv
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# place the dataset
# download from: https://www.kaggle.com/datasets/namanj27/ner-dataset
mkdir -p data/raw
# move ner_dataset.csv into data/raw/

# preprocess
python -m pipeline.data.preprocess

# train (pick one or all)
python -m pipeline.training.train --model crf
python -m pipeline.training.train --model bilstm_crf
python -m pipeline.training.train --model bert_ner
python -m pipeline.training.train --model distilbert_ner

# evaluate and promote best
python -m pipeline.training.evaluate

# run the api
uvicorn api.main:app --reload

# frontend (separate terminal)
cd web
npm install
npm run dev
```

## GPU

BiLSTM-CRF and BERT-NER use GPU automatically when available. If `torch.cuda.is_available()` returns `False`, you likely have the CPU-only PyTorch wheel:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## API

| Endpoint | Method | What |
|---|---|---|
| `/api/ner` | POST | Run NER on text, get tokens + tags + entities |
| `/api/translate` | POST | NER-aware translation |
| `/api/ws/translate` | WS | Realtime — streams NER + translation as you type |
| `/api/entities` | GET | All collected entities this session |
| `/api/entities` | DELETE | Clear collected entities |
| `/health` | GET | Liveness check |
| `/metrics` | GET | Prometheus metrics |

**Request body** for `/api/translate`:
```json
{ "text": "Shah Jahan built the Taj Mahal", "target_lang": "hi" }
```

**Response**:
```json
{
  "source_text": "Shah Jahan built the Taj Mahal",
  "translated_text": "Shah Jahan ने Taj Mahal बनवाया",
  "entities": [
    { "text": "Shah Jahan", "label": "per", "start": 0, "end": 10 },
    { "text": "Taj Mahal", "label": "geo", "start": 21, "end": 30 }
  ],
  "target_lang": "hi"
}
```

## Translation backends

Set `TRANSLATION_BACKEND` env var:

| Backend | Value | Needs |
|---|---|---|
| Google Translate | `google` (default) | `deep-translator` (included) |
| MyMemory Hub | `mymemory` | Free collaborative TM database |
| MarianMT | `marian` | Downloads Helsinki-NLP models locally |

### Robust Restoration
Different translators handle markers differently. **Turing Tag** uses a fuzzy recovery engine that can identify and restore markers even if they've been mangled with spaces or casing changes (e.g., `__ENT0__` → `__ ENT 0 __`).

### Lexicon Gazetteer
If the machine learning models miss a common entity (like "Google"), you can add it to `data/gazetteer.json`. The system will automatically detect these as high-confidence entities before translation.

Adding a new backend: implement `api/services/translation/base.py::Translator` and register it in `factory.py`.

## MLflow

Experiments log to a local SQLite database. To view:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Opens a dashboard at `localhost:5000` where you can compare runs.

## DVC

`dvc.yaml` defines the full reproducible pipeline: preprocess → train × 4 → evaluate. Run it end-to-end with:

```bash
dvc repro
```

## Tests

```bash
pytest tests/ -v
```

18 tests covering: data loading, vocab construction, BiLSTM-CRF forward/backward + convergence, CRF feature extraction, POS guesser, entity masking/restoration, response structure, API endpoints.

## Data split

Train / Val / Test = 70 / 15 / 15 of the GMB sentences, deterministic via `random_state=42` in `pipeline/data/preprocess.py`. Split is computed in two stages: 15% test off the top, then 15% of the original as validation from what remains. Same split is used across all four models for apples-to-apples comparison.

## CI

GitHub Actions runs on every push and PR to `main`:

- **Lint** — flake8 across pipeline, api, tests
- **Pipeline tests** — data processing, model shapes, feature extraction
- **API tests** — endpoint health, translation service logic
- **Frontend** — TypeScript type-check + Vite production build
- **Train** (main only) — preprocess → train CRF → evaluate → upload artifacts

## Project structure

```
turing_tag/
├── .github/workflows/ci.yml
├── pipeline/
│   ├── config.py
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── crf_model.py
│   │   ├── bilstm_crf.py
│   │   ├── bert_ner.py
│   │   └── distilbert_ner.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── registry/
│       └── promote.py
├── api/
│   ├── main.py
│   ├── deps.py
│   ├── routes/
│   │   ├── ner.py
│   │   ├── translate.py
│   │   └── stream.py
│   ├── services/
│   │   ├── ner_service.py
│   │   ├── translate_service.py
│   │   └── translation/
│   │       ├── base.py
│   │       ├── factory.py
│   │       ├── google.py
│   │       └── marian.py
│   └── schemas/
│       └── models.py
├── web/                         React + TS + Vite
├── monitoring/prometheus.yml
├── tests/
├── params.yaml
├── dvc.yaml
└── requirements.txt
```

## License

MIT
