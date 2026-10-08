# Dataset research and selection

Research date: **7 October 2026**. Search terms covered temporal validity, information obsolescence, temporal claims, freshness, recency and time-sensitive factual statements. Exa search reviewed 18 result entries across three searches, followed by inspection of author repositories, dataset cards and raw data. Only the public dataset preparation stage uses the network; the classifier never searches for the truth of an input.

| Candidate | Relevant annotation | Fit for this project | Decision |
|---|---|---|---|
| [EverGreenQA](https://huggingface.co/datasets/s-nlp/EverGreen-Multilingual), [author repository](https://github.com/s-nlp/Evergreen), [paper](https://arxiv.org/abs/2505.21115) | Evergreen versus mutable factual questions; English plus six other languages. | Closest verified binary target, but inputs are questions rather than claims. Author CSVs also include English training answers. | Primary auxiliary dataset, with statement adaptation. |
| [SituatedQA](https://situatedqa.github.io/), [paper](https://aclanthology.org/2021.emnlp-main.586/) | Context-dependent question identification, including binary temporal dependence. | Relevant question task; QA triplet exports should not be mistaken for the identification labels. | Useful alternative; not mixed into this baseline. |
| [FreshQA](https://github.com/freshllms/freshqa), [paper](https://arxiv.org/abs/2310.03214) | Never-changing, slow-changing and fast-changing questions; also false-premise and answer evaluation fields. | Change-rate labels could be collapsed, but truth/false-premise fields are a different target. | Alternative evaluation/adaptation source. |
| [TempLAMA](https://arxiv.org/abs/2106.15110) | Time-varying cloze queries with temporally scoped answers. | Useful volatile examples; not a ready balanced stable/volatile claim dataset. | Not used for this baseline. |
| [WhenFactsExpire](https://zenodo.org/records/17201109) | Temporal plausibility of knowledge-graph facts with corrupted time scopes. | Checks validity at a time, which differs from predicting whether an information type can change. | Rejected for the target label. |
| [RecencyQA](https://github.com/DataScienceUIBK/RecencyQA) | Expected update frequencies, including “Never,” plus stationarity. | Relevant alternative, but question-based and substantially model-annotated rather than a direct claim benchmark. | Not used; review as future work. |

No ready-to-use binary **answer-sentence** dataset with the exact project policy was verified in this search. This is a bounded search result, not a claim that none exists. A suitable binary **question** dataset was found, so the baseline uses that resource and explicitly reports the transfer limitation.

## Selected source and mapping

Hugging Face dataset: `s-nlp/EverGreen-Multilingual`.

- Dataset revision: `0a114a0126272e991a368dddcbe3092bd72dbd16`.
- Author GitHub revision used for CSV downloads: `88f9759fbf66abfd45c454fe4d9d2db20cd6a5da`.
- Observed source splits: 2,008 `train`, 1,479 `train_synth`, 1,270 `test`.
- Source `is_evergreen=1` means stable; `is_evergreen=0` means mutable.
- Project conversion is **`label = 1 - is_evergreen`**.

Raw row counts were read from actual files. The paper's description of synthetic data differs from the current `train_synth` file: the downloaded file contains 1,449 mutable and 30 evergreen questions. The loader follows each row's label, and asserts unmistakable examples to verify the mapping. It does not assign labels based on a split's name or the paper's aggregate description. The Hugging Face card does not explicitly declare a dataset license; the repository README's MIT badge alone is not treated as a dataset license declaration. Source provenance is preserved, and raw files are gitignored.

## Statement adaptation

The author training CSV contains `en_Answer`. The preparation script keeps a bounded subset of short single-sentence answers and uses their question labels as **weak supervision**. It filters retrospective wording from volatile examples because a changing question can have a historical stable sentence in its answer. This filter reduces one obvious failure mode; it is not a complete semantic annotation system.

The assistant also wrote 260 development statements and 100 frozen test statements, covering concepts, history, software, roles, fees, rules, schedules, rates, availability and statistics. Related development examples share groups. These statements include fixed-past/current minimal contrasts, false but stable claims and “current” as a physics term. They were labeled using [the project rubric](annotation_guide.md). They are **not independently human-labeled** and their illustrative factual values were not checked for truth.

After filtering, deduplication and group splitting, the initial dataset had 3,159 training rows and 785 validation rows. A reported short-forecast failure led to 64 additional assistant-authored training regressions, appended after splitting: the processed manifest now records 3,223 training rows, including 270 authored statements. Validation still has 785 rows, including 54 authored statements. The source's 1,270 test questions and the 100 authored test statements remain unchanged. The new rows cover changeable forecasts alongside stable historical snapshots, definitions and enduring scientific relations; they are regression examples, not new benchmark evidence. All six pairwise split checks find zero exact normalized text overlap and zero group overlap. Questions and answers share source groups. The split audit does not guarantee semantic independence across separately authored claims.

Each processed row records `id`, `text`, `label`, `group`, `source`, `kind`, `annotation_status`, `source_question` and `source_url`. The manifest stores source checksums and the preparation policy. Training consumes these local files without accessing the internet.

## Why not Kaggle or a large model?

The verified Hugging Face/author source matches the auxiliary classification target. Kaggle CPU execution remains available through the included notebook, but no hosted job is needed. The classifier now combines a fixed pretrained sentence encoder with a supervised temporal head; see [semantic research](semantic_research.md). Encoder pretraining is separate from the measured local head fit. The executable project does not call an LLM to label data; project-authored examples were supplied and labeled by the coding assistant during development.
