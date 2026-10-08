import socket
from pathlib import Path

import pytest

from flagger.predict import StalenessClassifier
from flagger.model import make_pipeline


@pytest.fixture(scope="module")
def classifier():
    return StalenessClassifier()


@pytest.mark.parametrize("text,label", [
    ("The Earth orbits the Sun.", 0),
    ("Python was created by Guido van Rossum.", 0),
    ("Python is a programming language.", 0),
    ("The current price is €50.", 1),
    ("The latest Python version is 3.13.", 1),
    ("The current CEO is John Smith.", 1),
    ("The visa fee is €50.", 1),
    ("The visa fee was €50 in 2010.", 0),
    ("An electric current is a flow of electric charge.", 0),
    # This is false, but temporally stable: correctness is not the target.
    ("The Earth is shaped like a cube.", 0),
])
def test_required_examples_and_label_semantics(classifier, text, label):
    assert classifier.predict_claims([text])[0]["label"] == label


def test_inference_works_with_network_connections_blocked(classifier, monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("Classifier attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    result = classifier.analyze("The Earth orbits the Sun. The current price is €50.")
    assert result["summary"] == {"total": 2, "stable": 1, "volatile": 1}


def test_threshold_controls_decision_not_probability(classifier):
    text = "The university ranks tenth in the national league table."
    low = classifier.predict_claims([text], 0.1)[0]
    high = classifier.predict_claims([text], 0.9)[0]
    assert low["volatility_probability"] == high["volatility_probability"]
    assert low["label"] >= high["label"]


@pytest.mark.parametrize("text,label", [
    ("euro will go up", 1),
    ("The euro will go up.", 1),
    ("The euro will rise.", 1),
    ("The dollar is going to go down.", 1),
    ("The price of gold will rise.", 1),
    ("The euro was introduced in 1999.", 0),
    ("The euro went up in 2009.", 0),
    ("Gold went down in 2013.", 0),
    ("A triangle will always have three sides.", 0),
    ("The Sun will rise tomorrow.", 0),
])
def test_forecasts_and_stable_temporal_contrasts(classifier, text, label):
    # Reported cases and related checks are regressions, not accuracy evidence.
    assert classifier.predict_claims([text])[0]["label"] == label


def test_empty_batch_and_limits(classifier):
    assert classifier.predict_claims([]) == []
    with pytest.raises(ValueError, match="at most"):
        classifier.analyze("A claim. " * 151)
    with pytest.raises(ValueError, match="Threshold"):
        classifier.predict_claims(["A claim."], 1)


def test_missing_model_reports_training_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/train.py"):
        StalenessClassifier(tmp_path / "missing.joblib")


def test_training_pipeline_is_supervised_and_download_free():
    # Tiny arbitrary labels prove this pipeline learns from data, without a
    # knowledge lookup, pretrained checkpoint, or hard-coded decision rules.
    model = make_pipeline("word_logistic")
    model.fit(["apples grow", "apples ripen", "tickets cost", "tickets expire"], [0, 0, 1, 1])
    assert model.predict(["apples grow", "tickets expire"]).tolist() == [0, 1]
