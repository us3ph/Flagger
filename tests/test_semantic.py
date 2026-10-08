"""Integration checks for offline semantic assets and movable model bundles."""

import csv
import json
import os
from pathlib import Path
import shutil
import socket

import pytest

from flagger.predict import DEFAULT_MODEL, StalenessClassifier
from flagger.semantic import get_encoder
from flagger.text import normalize_text

ROOT = Path(__file__).resolve().parents[1]


def test_cold_semantic_model_load_and_prediction_work_offline(monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("Local model attempted a network connection")

    get_encoder.cache_clear()
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    classifier = StalenessClassifier()
    assert classifier.metadata["semantic_encoder"]["frozen"] is True
    result = classifier.analyze("euro will go up. The euro was introduced in 1999.")
    assert [item["label"] for item in result["claims"]] == [1, 0]
    assert result["summary"] == {"total": 2, "stable": 1, "volatile": 1}


def test_model_and_encoder_can_move_together(tmp_path):
    artifact = tmp_path / "staleness.joblib"
    shutil.copyfile(DEFAULT_MODEL, artifact)
    # Hardlinks avoid copying 92 MB; loading still uses the moved directory.
    shutil.copytree(DEFAULT_MODEL.parent / "semantic_encoder", tmp_path / "semantic_encoder", copy_function=os.link)
    classifier = StalenessClassifier(artifact)
    branch = dict(classifier.pipeline.named_steps["features"].transformer_list)["semantic"]
    assert Path(branch.model_dir) == tmp_path / "semantic_encoder"
    assert classifier.predict_claims(["euro will go up"])[0]["label"] == 1


def test_missing_encoder_is_reported_without_downloading(tmp_path, monkeypatch):
    artifact = tmp_path / "staleness.joblib"
    shutil.copyfile(DEFAULT_MODEL, artifact)
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("Unexpected download"))
    with pytest.raises(FileNotFoundError, match="train_semantic.py"):
        StalenessClassifier(artifact)


def test_paraphrase_diagnostic_is_separate_from_existing_splits():
    known = set()
    for split in ["train", "validation", "test_questions", "test_statements"]:
        with (ROOT / "data/processed" / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            known.update(normalize_text(row["text"]) for row in csv.DictReader(handle))
    with (ROOT / "data/paraphrase_check.tsv").open(newline="", encoding="utf-8") as handle:
        diagnostic = list(csv.DictReader(handle, delimiter="\t"))
    assert not known & {normalize_text(row["text"]) for row in diagnostic}
    assert len(diagnostic) == 60
    assert {row["label"] for row in diagnostic} == {"0", "1"}
    report = json.loads((ROOT / "reports/metrics.json").read_text())
    assert "not independently human-reviewed" in report["paraphrase_note"]
