# Evaluation report

Selection uses validation statements only. Both test sets remain outside training and selection. The saved classifier is the evaluated training-only model; validation is not folded into it.

## Validation comparison

| Model | Train seconds | Statement macro-F1 | All-data macro-F1 |
|---|---:|---:|---:|
| word_logistic | 0.099 | 0.926 | 0.942 |
| word_char_logistic | 0.507 | 0.907 | 0.936 |
| word_naive_bayes | 0.079 | 0.812 | 0.928 |

Selected: **word_logistic**.

## Held-out evaluation

| Set | N | Accuracy | Macro-F1 | Volatile precision | Volatile recall | Majority macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| questions | 1270 | 0.860 | 0.818 | 0.714 | 0.749 | 0.427 |
| statements | 100 | 0.840 | 0.840 | 0.854 | 0.820 | 0.333 |

### Questions confusion matrix

Rows are gold labels; columns are predictions.

| | Predicted stable | Predicted volatile |
|---|---:|---:|
| Gold stable | 850 | 97 |
| Gold volatile | 81 | 242 |

### Statements confusion matrix

Rows are gold labels; columns are predictions.

| | Predicted stable | Predicted volatile |
|---|---:|---:|
| Gold stable | 43 | 7 |
| Gold volatile | 9 | 41 |

## Interpretation

The official test contains questions, so its score does not measure answer-sentence performance. The statement test is a small assistant-authored exercise labeled with the project rubric; it is not an independently human-annotated benchmark. Both question-to-statement transfer and annotation bias limit generalization. Short source answers are weakly labeled and may contain noise. Exact text and source/family groups are disjoint across splits; semantic overlap between independently written examples can still exist.

Inspect `errors.csv` for every test error and `top_features.json` for learned lexical associations. The system labels volatility, including potentially false claims; it never verifies correctness.

Probabilities are not calibrated. Timing is a local CPU measurement and excludes HTTP and startup overhead.
