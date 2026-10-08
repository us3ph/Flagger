# Evaluation report

Semantic settings selected by macro-F1 on existing validation statements only; test and paraphrase labels excluded. Both test sets remain outside training and selection. The saved classifier is the evaluated training-only model; validation is not folded into it.

## Validation comparison

| Model | Train seconds | Statement macro-F1 | All-data macro-F1 |
|---|---:|---:|---:|
| word_logistic | 0.944 | 0.926 | 0.942 |
| hybrid_logistic | 23.682 | 0.944 | 0.956 |
| semantic_logistic | 23.261 | 0.833 | 0.899 |

Selected: **hybrid_logistic**. Validation winner: **hybrid_logistic**.

## Held-out evaluation

| Set | N | Accuracy | Macro-F1 | Volatile precision | Volatile recall | Majority macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| questions | 1270 | 0.869 | 0.833 | 0.715 | 0.802 | 0.427 |
| statements | 100 | 0.950 | 0.950 | 0.959 | 0.940 | 0.333 |

### Questions confusion matrix

Rows are gold labels; columns are predictions.

| | Predicted stable | Predicted volatile |
|---|---:|---:|
| Gold stable | 844 | 103 |
| Gold volatile | 64 | 259 |

### Statements confusion matrix

Rows are gold labels; columns are predictions.

| | Predicted stable | Predicted volatile |
|---|---:|---:|
| Gold stable | 48 | 2 |
| Gold volatile | 3 | 47 |

## Interpretation

The official test contains questions, so its score does not measure answer-sentence performance. The statement test is a small assistant-authored exercise labeled with the project rubric; it is not an independently human-annotated benchmark. Both question-to-statement transfer and annotation bias limit generalization. Short source answers are weakly labeled and may contain noise. Exact text and source/family groups are disjoint across splits; semantic overlap between independently written examples can still exist.

Inspect `errors.csv` for every test error and `top_features.json` for learned lexical associations. The system labels volatility, including potentially false claims; it never verifies correctness.

Probabilities are not calibrated. Timing is a local CPU measurement and excludes HTTP and startup overhead.

## Semantic comparison

A fixed, pretrained MiniLM encoder supplies contextual sentence vectors. A supervised classifier combines these with lexical cues that help preserve tense and historical anchoring. The encoder is not fine-tuned on this project. The one-second lexical fitting time did not include any semantic pretraining.

| Model | Statement accuracy | Statement macro-F1 | Paraphrase accuracy |
|---|---:|---:|---:|
| word_logistic | 0.840 | 0.840 | 0.817 |
| hybrid_logistic | 0.950 | 0.950 | 0.883 |

60 assistant-authored, rubric-labeled statements grouped into 20 paraphrase families; not independently human-reviewed, not used for model selection. These figures evaluate assigned temporal labels, not factual truth. Inspect all remaining failures in `errors.csv`.
