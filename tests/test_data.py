import csv
import hashlib
import json

from flagger.predict import ROOT, StalenessClassifier
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


def test_reported_forecast_is_training_only():
    train = load("train")
    regressions = [r for r in train if r["id"].startswith("curated_regressions_")]
    assert regressions
    assert any(r["text"] == "euro will go up" and r["label"] == "1" for r in regressions)
    assert {r["label"] for r in regressions} == {"0", "1"}
    assert all("human_review" in r["annotation_status"] for r in regressions)
    for name in ["validation", "test_questions", "test_statements"]:
        assert not any(r["id"].startswith("curated_regressions_") for r in load(name))


def test_saved_model_matches_current_training_data():
    metadata = StalenessClassifier().metadata
    assert metadata["training_rows"] == len(load("train"))
    training_bytes = (ROOT / "data/processed/train.csv").read_bytes()
    assert metadata["data_sha256"] == hashlib.sha256(training_bytes).hexdigest()
