# Annotation guide

Label the **claim as worded**, not whether you believe its content.

Ask: **Could the information asserted here become outdated as time passes?** Assume the statement refers to the present unless it explicitly describes a completed historical event or a fixed past time.

| Label | Definition | Common examples |
|---|---|---|
| 0 — STABLE | A definition, basic concept, enduring scientific relation or completed historical assertion. | Authorship, founding dates, mathematical relations, an explicitly dated past price. |
| 1 — TIME-SENSITIVE | A present status, value, role, availability, rule, schedule, record, ranking or future plan whose information may change. | Prices, fees, officeholders, software releases, membership, eligibility, forecasts. |

“Stable” is a practical evergreen category, not a promise of eternal truth. Basic scientific claims may be corrected by new research, but label them stable under this project's practical definition. Present-day institutional rules and officeholders are volatile even if they have not changed recently. The frequency of change is not modeled.

## Resolve common traps

- **Truth is independent.** “A triangle has four sides” is false but stable. “The current price is €999999” is volatile even if fabricated.
- **Temporal anchoring matters.** “The fee is €50” is volatile; “The fee was €50 in 2010” is stable. “As of today, the fee is €50” remains volatile because “today” moves. This project treats an explicitly fixed historical snapshot as stable.
- **A year alone is insufficient.** “The next election will be in 2030” is a changeable plan, whereas “The election took place in 2010” is completed history.
- **Definitions differ from present instances.** “A CEO manages a company” is stable; “Alice is the CEO” is volatile. “Current is measured in amperes” is stable.
- **Rates and amounts are not always volatile.** Unit conversions and chemical constants are stable. A quoted subscription fee is volatile.
- **Latest and records are moving targets.** A named winner of the 2014 tournament is stable. A reigning champion or current record holder is volatile.
- **Negation does not change the task.** “The item is not available” is still a changeable availability statement.
- **Subjective or non-factual text is out of scope.** A question, command, personal opinion or ambiguous fragment should be marked for review, rather than forced into a gold claim dataset. The demo baseline will still return a binary prediction for prose units it receives.

## Annotation procedure for the BSc submission

1. Give two annotators the same frozen rubric and a manageable claim sample.
2. Record each annotator's label independently, without checking current truth.
3. Preserve sentence IDs, original wording, source question, adaptation notes and any uncertainty.
4. Calculate agreement, such as Cohen's kappa, before adjudication.
5. Discuss disagreements; record adjudicated labels and rationales.
6. Group related claims and paraphrases before splitting. Freeze a test set before selecting features, models or thresholds.

The current `curated_train.tsv` and `curated_test.tsv` contain **assistant-authored** labels. No independent human review, inter-annotator agreement study or claim-level gold annotation has been performed. Public short answers use weaker inherited question labels; do not treat them as manually validated claim labels. For mixed stable/volatile sentences, extract and annotate atomic claims manually when building a more rigorous dataset.
