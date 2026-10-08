# BSc project methodology

## Title and research question

**AI Answer Staleness Flagger**

Can a small text classifier identify sentences whose asserted information is likely to change over time, without verifying facts or using external retrieval?

The target is temporal volatility, not factual truth, actual expiry, or a prediction of when an update will occur. Output is `0 = STABLE` or `1 = TIME-SENSITIVE / VOLATILE`.

## Objectives

1. Locate a public dataset with relevant temporal labels and document its fit.
2. Prepare a small statement adaptation with traceable labels and sources.
3. Compare lexical classifiers with fixed pretrained semantic features on CPU.
4. Evaluate on held-out questions and separately on held-out statements.
5. Demonstrate sentence-level flagging in a usable local application.

## Pipeline

```mermaid
flowchart LR
    A[Paste an AI answer] --> B[Split sentence spans]
    B --> C[Normalize text]
    C --> D[Word TF-IDF features]
    C --> H[Fixed MiniLM sentence embeddings]
    D --> E[Supervised temporal classifier]
    H --> E
    E --> F[0 Stable / 1 Volatile]
    F --> G[Monochrome highlights and verification counts]
```

The splitter preserves original offsets for display, common abbreviations, decimal prices and version numbers. Newlines and semicolons define additional units; headings and fenced code are omitted. It is deliberately a sentence baseline. An atomic fact extractor would be a separate, more complex component.

Normalization applies Unicode normalization, quote normalization, lowercase and whitespace normalization. It keeps negation, tense words, historical years, release cues and ordinary stopwords. Removing words such as “current,” “latest,” “was” or “not” would discard useful temporal information.

Word TF-IDF uses unigrams and bigrams, sublinear term frequency and a maximum of 18,000 vocabulary features. A comparison adds character n-grams of lengths 3–5, capped at 22,000 additional features. TF-IDF is fitted on training text only.

The semantic model adds 384-dimensional normalized sentence vectors from a pinned all-MiniLM-L6-v2 checkpoint. Its pretrained encoder weights stay fixed. A fixed validation grid compares semantic-only and combined features with embedding weights 0.5, 1, 2 and 4, and Logistic Regression `C` values 1, 4 and 16. Balanced class weights, `liblinear` and random seed 42 are fixed. Authored training statements receive sample weight 4 to emphasize the application format; auxiliary questions and weak answers have weight 1. Validation selects a combined head with `C=4` and embedding weight 0.5. The original lexical comparison remains in `scripts/train.py`; the semantic experiment is in `scripts/train_semantic.py`.

## Experimental protocol

- Use the pinned English EverGreenQA source with the verified inverted label mapping.
- Deduplicate training texts and exclude conflicting duplicate labels.
- Reserve official source test questions and the authored statement test before fitting.
- Split development rows 80/20 by source/family group using seed 42. Source questions and adapted answers remain together.
- Keep user examples in training as regression examples, outside accuracy evidence.
- Append the 64 short-forecast regression statements after the original split. Keep validation and test membership unchanged; audit all split pairs again.
- Fit lexical, semantic-only and combined classification heads. Choose semantic settings using only macro-F1 on the existing 54 authored validation statements. Do not tune on either test set.
- Evaluate a separate 60-statement paraphrase diagnostic after selection. Its assistant-authored labels and exact texts never participate in fitting or selection; report its annotation and topic-overlap limitations.
- Evaluate the selected model at threshold 0.5, and retain the same training-only artifact for the app.
- Report accuracy, macro-F1, volatile precision/recall, confusion matrices, a majority baseline, local training time, artifact size and inference timing.
- Preserve all test errors for qualitative review.

Macro-F1 gives both labels equal weight, even when the official question test contains more stable examples. Volatile recall measures how often changing information is flagged; precision measures how often flags match the assigned labels. Neither metric measures factual correctness.

## Results and analysis

The selected classifier combines MiniLM and word TF-IDF features, fitted on all 3,223 training rows. The measured question accuracy is 86.9% and macro-F1 is 0.833. On the assistant-authored 100-statement exercise, accuracy is 95.0% and macro-F1 is 0.950, compared with 84.0% for the lexical baseline. The paraphrase exercise improves from 49/60 to 53/60 correct, with seven errors retained for review. The complete report is [../reports/evaluation.md](../reports/evaluation.md).

The classifier artifact is about 264 KiB, plus approximately 92 MB of encoder assets. Preparing sentence features took 22.88 seconds and fitting the selected head took 0.80 seconds on the available CPU; the full comparison took about 54 seconds. The encoder's earlier pretraining is not included in these measurements. `top_features.json` exports lexical coefficients only; embedding dimensions do not correspond to individual words. The authored evaluations are small and annotation-dependent and do not validate general-purpose accuracy.

Discuss representative errors from `reports/errors.csv`: indirect present roles, availability without freshness vocabulary, definitions containing volatile-domain words and historical assertions with prices or offices. Public question labels can also reflect a different policy from this claim rubric. Do not silently relabel the official test to improve results.

## Application and verification

FastAPI loads a trusted local artifact at startup and checks for completed file replacements before each API request. Training writes to a temporary file and atomically replaces the artifact after saving succeeds. A lock serializes reloads; each request receives a fully loaded classifier instance. It batches the sentence predictions and returns labels, scores, offsets, counts and timing. Browser rendering uses original character spans and text nodes, preserving Unicode and escaping pasted HTML. Displayed counts, highlights and exports use the same 0/1 predictions. Inference has no web-search dependency, API key or external model service.

Verification covers label semantics and required examples, fixed-past/current contrasts, false stable claims, segmentation, leakage audits, input limits and inference with network connections blocked. Browser checks exercise desktop and mobile layouts, label highlighting, sentence details, copy/export, Unicode and HTML text escaping.

## Limitations and future work

The main limitation is training on questions while applying the model to sentences. Weak source answers and assistant-authored annotations introduce noise and bias. Independent annotators should review and expand the statement data using a frozen rubric and report agreement before adjudication.

Other limitations include English-only inputs, small statement evaluation, mixed claims inside one sentence, pronouns, ambiguous temporal scope, unstable institutional conventions and uncalibrated scores. The encoder truncates at 256 wordpieces. Semantic features improve the measured behavior but do not guarantee paraphrase consistency. A binary target treats an hourly price and a rarely changing institutional rule alike.

Useful extensions are a larger independently annotated statement benchmark, grouped cross-validation on development data, probability calibration on an additional validation set and improved segmentation. Encoder fine-tuning would be a separate experiment requiring suitable data and another untouched evaluation. See [semantic research notes](semantic_research.md) for the distinction between this implementation and the MELD AI authorship detector.
