# BSc project methodology

## Title and research question

**AI Answer Staleness Flagger**

Can a small text classifier identify sentences whose asserted information is likely to change over time, without verifying facts or using external retrieval?

The target is temporal volatility, not factual truth, actual expiry, or a prediction of when an update will occur. Output is `0 = STABLE` or `1 = TIME-SENSITIVE / VOLATILE`.

## Objectives

1. Locate a public dataset with relevant temporal labels and document its fit.
2. Prepare a small statement adaptation with traceable labels and sources.
3. Train supervised CPU classifiers with simple, explainable text features.
4. Evaluate on held-out questions and separately on held-out statements.
5. Demonstrate sentence-level flagging in a usable local application.

## Pipeline

```mermaid
flowchart LR
    A[Paste an AI answer] --> B[Split sentence spans]
    B --> C[Normalize text]
    C --> D[TF-IDF features]
    D --> E[Small supervised classifier]
    E --> F[0 Stable / 1 Volatile]
    F --> G[Green / red highlights]
```

The splitter preserves original offsets for display, common abbreviations, decimal prices and version numbers. Newlines and semicolons define additional units; headings and fenced code are omitted. It is deliberately a sentence baseline. An atomic fact extractor would be a separate, more complex component.

Normalization applies Unicode normalization, quote normalization, lowercase and whitespace normalization. It keeps negation, tense words, historical years, release cues and ordinary stopwords. Removing words such as “current,” “latest,” “was” or “not” would discard useful temporal information.

Word TF-IDF uses unigrams and bigrams, sublinear term frequency and a maximum of 18,000 vocabulary features. A comparison adds character n-grams of lengths 3–5, capped at 22,000 additional features. TF-IDF is fitted on training text only.

Logistic Regression uses `C=4`, balanced class weights and the `liblinear` solver. The Naive Bayes comparison uses `alpha=0.5`. These settings are fixed before the evaluation. Authored development statements receive sample weight 4 to emphasize the application format; auxiliary questions and weak answers have weight 1. There is no transformer model, checkpoint download, language-model training or fine-tuning.

## Experimental protocol

- Use the pinned English EverGreenQA source with the verified inverted label mapping.
- Deduplicate training texts and exclude conflicting duplicate labels.
- Reserve official source test questions and the authored statement test before fitting.
- Split development rows 80/20 by source/family group using seed 42. Source questions and adapted answers remain together.
- Keep user examples in training as regression examples, outside accuracy evidence.
- Train three small candidates and choose the highest macro-F1 on authored validation statements. Do not tune on either test set.
- Evaluate the selected model at threshold 0.5, and retain the same training-only artifact for the app.
- Report accuracy, macro-F1, volatile precision/recall, confusion matrices, a majority baseline, local training time, artifact size and inference timing.
- Preserve all test errors for qualitative review.

Macro-F1 gives both labels equal weight, even when the official question test contains more stable examples. Volatile recall measures how often changing information is flagged; precision measures how often flags match the assigned labels. Neither metric measures factual correctness.

## Results and analysis

The selected classifier is word TF-IDF + Logistic Regression. The current measured question accuracy is 86.0% and macro-F1 is 0.818. On the assistant-authored 100-statement exercise, accuracy and macro-F1 are 0.840. The complete report is [../reports/evaluation.md](../reports/evaluation.md).

The model is about 253 KiB and trained in approximately 0.1 seconds on the available CPU. The artifact's learned feature coefficients are exported in `top_features.json`. These explain lexical associations, not a causal semantic understanding of time. The statement test is small and annotation-dependent; the results support a functioning BSc baseline, not a validated general-purpose deployment claim.

Discuss representative errors from `reports/errors.csv`: indirect present roles, availability without freshness vocabulary, definitions containing volatile-domain words and historical assertions with prices or offices. Public question labels can also reflect a different policy from this claim rubric. Do not silently relabel the official test to improve results.

## Application and verification

FastAPI loads one trusted local artifact at startup. It batches the sentence predictions and returns labels, scores, offsets, counts and timing. Browser rendering uses original character spans and text nodes, preserving Unicode and escaping pasted HTML. Inference has no web-search dependency, API key or external model service.

Verification covers label semantics and required examples, fixed-past/current contrasts, false stable claims, segmentation, leakage audits, input limits and inference with network connections blocked. Browser checks exercise desktop and mobile layouts, label highlighting, sentence details, copy/export, Unicode and HTML text escaping.

## Limitations and future work

The main limitation is training on questions while applying the model to sentences. Weak source answers and assistant-authored annotations introduce noise and bias. Independent annotators should review and expand the statement data using a frozen rubric and report agreement before adjudication.

Other limitations include English-only inputs, small statement evaluation, mixed claims inside one sentence, pronouns, ambiguous temporal scope, unstable institutional conventions and uncalibrated scores. Lexical classification cannot guarantee semantic understanding. A binary target treats an hourly price and a rarely changing institutional rule alike.

Useful extensions are a larger independently annotated statement benchmark, grouped cross-validation on development data, probability calibration on an additional validation set, an optional small **fixed embedding** feature comparison with a linear classifier, and improved deterministic segmentation. These extensions are not required to demonstrate the current project and should not involve fine-tuning a language model.
