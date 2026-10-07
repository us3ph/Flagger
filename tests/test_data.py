import csv
import json

from flagger.predict import ROOT
from flagger.text import normalize_text


def load(name):
    with (ROOT / "data/processed" / f"{name}.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_no_text_or_source_group_leakage():
    splits = {name: load(name) for name in ["train", "validation", "test_questions", "test_statements"]}
    names = list(splits)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            assert not ({normalize_text(r["text"]) for r in splits[name]} & {normalize_text(r["text"]) for r in splits[other]})
            assert not ({r["group"] for r in splits[name]} & {r["group"] for r in splits[other]})
    assert all(set(row["label"] for row in rows) == {"0", "1"} for rows in splits.values())


def test_dataset_mapping_provenance_and_statement_evaluation():
    manifest = json.loads((ROOT / "data/processed/manifest.json").read_text())
    assert manifest["mapping"]["upstream_is_evergreen_1"] == 0
    assert manifest["mapping"]["upstream_is_evergreen_0"] == 1
    assert all(row["kind"] == "statement" for row in load("test_statements"))
    assert all("human_review" in row["annotation_status"] for row in load("test_statements"))
    assert all(len(item["sha256"]) == 64 for item in manifest["downloads"])
