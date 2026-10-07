import pytest

from flagger.text import normalize_text, split_claims


@pytest.mark.parametrize("answer,expected", [
    ("The latest Python version is 3.13. The current price is €50.", ["The latest Python version is 3.13.", "The current price is €50."]),
    ("The price is $12.50. It is in stock.", ["The price is $12.50.", "It is in stock."]),
    ("Dr. Smith founded the company. The fee is €50.", ["Dr. Smith founded the company.", "The fee is €50."]),
    ("The train departs at 8:30 a.m. The fare is €20.", ["The train departs at 8:30 a.m.", "The fare is €20."]),
    ('He said "The fee is €50." Python is a language.', ['He said "The fee is €50."', "Python is a language."]),
    ("# Answer\n- Python is a language.\n2. The latest version is 3.13.\n```python\nx = 5\n```", ["Python is a language.", "The latest version is 3.13."]),
    ("A triangle has three sides; the current price is €50.", ["A triangle has three sides;", "the current price is €50."]),
    ("\n  Python is a language.\n\nThe fee is €50.  ", ["Python is a language.", "The fee is €50."]),
    ("🟢 Earth is a planet. 🔴 The fee is €50.", ["🟢 Earth is a planet.", "🔴 The fee is €50."]),
    ("", []),
    ("# Heading\n```\nnot a claim\n```", []),
])
def test_sentence_boundaries_and_original_offsets(answer, expected):
    spans = split_claims(answer)
    assert [span.text for span in spans] == expected
    assert all(answer[span.start:span.end] == span.text for span in spans)
    assert all(left.end <= right.start for left, right in zip(spans, spans[1:]))


def test_preprocessing_keeps_temporal_cues_negation_and_dates():
    assert normalize_text("  It is NOT the latest version in 2024. ") == "it is not the latest version in 2024."
