# AI Answer Staleness Flagger

A BSc project that predicts **whether a sentence's information could change over time**.

**0 = STABLE 🟢** · **1 = TIME-SENSITIVE / VOLATILE 🔴**

It combines a **pretrained MiniLM sentence encoder, TF-IDF temporal cues and a supervised Logistic Regression classifier**. Contextual sentence vectors help the model handle different wording. The encoder's weights stay fixed; the classifier learns stable/time-sensitive labels from this project's training data. Inference runs locally on CPU, without web search or fact verification.

```text
AI answer → sentences → MiniLM embeddings + TF-IDF → trained classifier → 0/1
```

The [MELD AI text detector](https://arxiv.org/abs/2605.06903) also uses a pretrained encoder and classification heads, but detects AI authorship. This project borrows that architectural idea and trains for temporal stability. It does not reproduce MELD's multitask training or reuse its AI/human labels. See [research and design notes](docs/semantic_research.md).

## Run the application

Python 3.11 or newer. The prepared data, trained classifier and local encoder assets are already present in this workspace. Starting the semantic model takes longer than the old lexical model because it loads neural weights.

```bash
cd /home/x-hunter/Flagger
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m uvicorn flagger.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Paste an English AI answer and click **Analyze**. The interface uses a black background, a custom FLAGGER wordmark and logo, and a compact editor with a word count. Model settings stay collapsed; results appear after analysis. The **Stability summary**, monochrome highlights, sentence details, decision threshold, copied labels and JSON export remain available. All inference runs on the local CPU. Inputs are not persisted by the application.

The summary reports counts of stable and time-sensitive claims, followed by a notice such as **“2 claims may need current verification.”** Counts are used only to identify what may need checking. Their ratio never rates an answer's truth, accuracy, reliability, or trustworthiness. When none are flagged, the interface still explains that stable claims can be incorrect. Labels are assigned per sentence, which may contain several claims.

Summary numbers are counts, not prediction labels. Counts, highlights and copied/exported summaries all come from the same returned 0/1 predictions.

On Windows, activate the environment with `.venv\Scripts\activate`. Run the commands from the project directory; replace the `cd` path if you move the project. The editable installation keeps the CLI connected to this source checkout and its model files. A fresh Git clone needs `python scripts/train_semantic.py` once before startup: the approximately 92 MB encoder directory is ignored by Git. That command downloads the pinned checkpoint if it is not cached, then trains the classifier. Runtime never downloads weights.

![Application preview](reports/app-desktop.png)

## Examples

| Claim | Label |
|---|---|
| The Earth orbits the Sun. | 🟢 0 |
| Python was created by Guido van Rossum. | 🟢 0 |
| Python is a programming language. | 🟢 0 |
| The latest Python version is 3.13. | 🔴 1 |
| The current price is €50. | 🔴 1 |
| The current CEO is John Smith. | 🔴 1 |
| The visa fee is €50. | 🔴 1 |
| The visa fee was €50 in 2010. | 🟢 0 |

An explicitly completed historical event stays stable even if its present-day equivalent changes. A definition such as “An electric current is a flow of electric charge” also stays stable. A false statement such as “The Earth is shaped like a cube” can receive **0**, because this classifier does not judge truth. Example prices, names and version numbers are illustrative; they were not checked for present-day accuracy.

## Dataset and training

The public source is [EverGreenQA / s-nlp/EverGreen-Multilingual](https://huggingface.co/datasets/s-nlp/EverGreen-Multilingual), associated with [Will It Still Be True Tomorrow?](https://arxiv.org/abs/2505.21115). It already has relevant binary labels for **questions**. This project reverses the source mapping: `is_evergreen=1 → STABLE=0`; `is_evergreen=0 → VOLATILE=1`.

For the answer-sentence task, it adds short source answers with explicitly **weak inherited labels**, plus 424 individually written project statements. These include 64 training regressions covering short forecasts, historical contrasts, definitions and enduring scientific statements. Project statements are **assistant-authored and rubric-labeled, not independently human-annotated**. Their labels and source-answer adaptations need human review for a final BSc research submission. Provenance and annotation status appear in each processed CSV row.

Training and validation have **3,223** and **785** examples. The classifier is fitted on every training row, with validation and test rows kept outside fitting. The two separate test sets are the **1,270 official questions** and **100 project-authored statements**. Exact normalized texts and question/family groups are disjoint across splits. Source questions and their answers stay in the same split. User-provided examples are training/regression cases, not accuracy evidence. The 64 new regression rows are appended after the original split so validation and both test datasets remain unchanged.

```bash
# Train the semantic model; download the pinned encoder only if needed.
python scripts/train_semantic.py

# Once encoder assets are present or cached, train without network access.
python scripts/train_semantic.py --offline

# Optional lexical baseline, written separately from the deployed model.
python scripts/train.py --model-kind word_logistic \
  --output-dir models/lexical_baseline --report-dir reports/lexical_baseline

# Optional: reproduce dataset preparation from pinned public source files.
python scripts/prepare_data.py

# Rebuild from cached raw sources without network access.
python scripts/prepare_data.py --offline
```

The downloader uses the pinned author GitHub repository because its CSV also supplies English answers omitted by the Hugging Face question dataset. It records revision IDs and SHA-256 checksums. Raw downloads stay in `data/raw/`. See [dataset research](docs/dataset_research.md) and the [annotation rubric](docs/annotation_guide.md).

The semantic training script compares a lexical baseline, a semantic-only classifier and a combined classifier. It evaluates a fixed grid of embedding weights and regularization strengths using the 54 authored validation statements. Validation selects the combined model with embedding weight 0.5 and `C=4`. The encoder is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with 384-dimensional sentence vectors. All 3,223 training rows fit the classification head; validation and test rows are never fitted. The original lexical comparison remains available in `scripts/train.py`.

The original model missed **“euro will go up”** because short forecast wording was poorly represented. It now receives **1**, without a keyword override. A separate 60-statement paraphrase diagnostic checks varied wording after model selection; it is never used for fitting or selection. Its statements are assistant-authored and are not an independent benchmark. Semantic features improve the measured results, but unfamiliar phrasing, opinions and incomplete fragments can still receive inappropriate labels.

Training writes the replacement model atomically. Once the updated server code is loaded, the app detects model-file replacements before its next API request, so subsequent retraining does not require a server restart. The model description refreshes after analysis, and browser responses disable caching. Changes to Python application code still require restarting the server.

## Measured results

| Held-out set | Accuracy | Macro-F1 | Volatile precision | Volatile recall |
|---|---:|---:|---:|---:|
| Official questions, n=1,270 | 86.9% | 0.833 | 0.715 | 0.802 |
| Project-authored statements, n=100 | 95.0% | 0.950 | 0.959 | 0.940 |

On the same 100-statement set, the lexical baseline scored **84%**; the semantic combination scored **95%**. On the 60-statement paraphrase diagnostic, results improved from **49/60 (81.7%)** to **53/60 (88.3%)**. Seven diagnostic errors remain and are recorded, including an implied gold-price forecast and some historical prices. These small assistant-authored checks do not establish general accuracy.

This CPU run spent **22.88 seconds** preparing sentence features and **0.80 seconds** fitting the selected classification head; the complete comparison took about **54 seconds**. A fast head fit is possible because semantic pretraining already happened before this project. The head artifact is about **264 KiB**, plus approximately **92 MB** of encoder assets; 20 short statements take about **69 ms** after loading. These are local measurements. Full comparisons, confusion matrices and errors are in [reports/evaluation.md](reports/evaluation.md), [reports/metrics.json](reports/metrics.json), and [reports/errors.csv](reports/errors.csv).

## CLI, API and verification

```bash
python -m flagger.cli 'Python is a programming language. The latest Python version is 3.13.'
python -m flagger.cli --json 'The current price is €50.'
python -m pytest -q
```

After `pip install -e .`, `staleness-flagger` is also available. Pipe an answer through stdin if preferred. API documentation is at `/docs` while the server runs.

```bash
curl http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"answer":"The Earth orbits the Sun. The visa fee is €50.","threshold":0.5}'
```

Tests verify required examples, decimal/version splitting, abbreviations, historical claims, label mapping, split leakage, limits, API behavior and inference with network connections blocked. Desktop/mobile browser checks also cover actual highlighting, Unicode offsets, copy/export and HTML text escaping; see [reports/verification.md](reports/verification.md).

## Project structure

```text
flagger/                 Preprocessing, model, inference, API, CLI
scripts/prepare_data.py   Pinned source download, adaptation, split audit
scripts/train_semantic.py Semantic comparison, training, evaluation
scripts/train.py          Original lexical classifier comparison
data/curated_*.tsv        Individually written statement examples and labels
data/processed/           Ready-to-train CSVs and provenance manifest
models/                  Trained head, metadata and local encoder assets
web/                     Browser interface; no external assets
tests/                   Meaningful behavior and offline-inference tests
notebooks/bsc_demo.ipynb  CPU training and demonstration, also usable on Kaggle
docs/                    Dataset research, annotation guide, BSc methodology
reports/                 Metrics, errors, learned features, app previews
```

See [docs/bsc_methodology.md](docs/bsc_methodology.md) for the research question, methods, experimental protocol and project limitations. Kaggle is optional: this classifier trains quickly on a laptop CPU, so no hosted training job or GPU is necessary. The notebook and [Kaggle instructions](docs/kaggle.md) are provided for reproducible student experiments.

## Scope and limitations

This English baseline uses sentences, newlines and semicolons as units. It preserves common abbreviations, decimal prices and version numbers, but it does not extract independent atomic facts from complex sentences, resolve pronouns, or reliably identify non-factual text. Markdown headings and fenced code are skipped. Mixed sentences receive one label and should be split manually for finer analysis.

The semantic encoder can miss implied volatility, while the lexical branch can overreact to familiar words or misread historical qualifiers. Sentence representations truncate after 256 wordpieces, so long mixed sentences can lose relevant context. Binary labels collapse different change rates. Scores are uncalibrated estimates of volatility, not probabilities of factual correctness, and **0 does not mean a claim is trustworthy**. Test errors remain visible in the report; improving the dataset and reviewing it with independent annotators are the next research steps.

Only load this project's trusted `joblib` model. To reproduce it on a different scikit-learn version, rerun training from the included CSVs.
