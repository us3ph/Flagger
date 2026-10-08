import pytest
from fastapi.testclient import TestClient

from flagger.app import app, ClassifierStore
from flagger.model import make_pipeline
from scripts.train import save_artifact


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_browser_assets_and_model_ready(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/health").json()["offline"] is True
    info = client.get("/api/model").json()
    assert info["fine_tuning"] is False
    assert info["labels"] == {"0": "STABLE", "1": "TIME-SENSITIVE"}


def test_actual_binary_predictions_and_spans(client):
    answer = "🟢 The Earth orbits the Sun.\n- The latest Python version is 3.13. The current price is €50."
    response = client.post("/api/analyze", json={"answer": answer})
    assert response.status_code == 200
    result = response.json()
    assert [r["label"] for r in result["claims"]] == [0, 1, 1]
    assert all(answer[r["start"]:r["end"]] == r["text"] for r in result["claims"])
    assert result["summary"] == {"total": 3, "stable": 1, "volatile": 2}


@pytest.mark.parametrize("payload,status", [
    ({"answer": "   "}, 400),
    ({"answer": "# Header\n```\nx = 1\n```"}, 400),
    ({"answer": ""}, 422),
    ({"answer": "x" * 25001}, 422),
    ({"answer": "A claim. " * 151}, 400),
    ({"answer": "A claim.", "threshold": 2}, 422),
    ({}, 422),
])
def test_input_validation(client, payload, status):
    assert client.post("/api/analyze", json=payload).status_code == status


def write_test_model(path, forecast_label, revision):
    model = make_pipeline("word_logistic")
    model.fit(["The euro will go up.", "A triangle has three sides."], [forecast_label, 1 - forecast_label])
    save_artifact({
        "format_version": 1,
        "pipeline": model,
        "metadata": {"model_kind": "word_logistic", "revision": revision},
    }, path)


def test_model_replacement_updates_predictions_and_metadata_without_restart(client, tmp_path, monkeypatch):
    path = tmp_path / "staleness.joblib"
    write_test_model(path, 0, "old")
    monkeypatch.setattr(app.state, "models", ClassifierStore(path))
    before = client.post("/api/analyze", json={"answer": "The euro will go up."}).json()
    assert before["claims"][0]["label"] == 0
    assert client.get("/api/model").json()["revision"] == "old"

    write_test_model(path, 1, "new")
    after = client.post("/api/analyze", json={"answer": "The euro will go up."}).json()
    assert after["claims"][0]["label"] == 1
    assert after["summary"] == {"total": 1, "stable": 0, "volatile": 1}
    assert client.get("/api/model").json()["revision"] == "new"
    assert client.get("/api/health").status_code == 200


def test_failed_model_replacement_is_reported_and_recovers(client, tmp_path, monkeypatch):
    path = tmp_path / "staleness.joblib"
    write_test_model(path, 1, "first")
    store = ClassifierStore(path)
    monkeypatch.setattr(app.state, "models", store)
    original = store.get()
    path.write_bytes(b"invalid model artifact")
    response = client.post("/api/analyze", json={"answer": "The euro will go up."})
    assert response.status_code == 503
    assert "could not be loaded" in response.json()["detail"]
    assert store._classifier is original
    write_test_model(path, 1, "repaired")
    assert client.get("/api/model").json()["revision"] == "repaired"
    assert client.post("/api/analyze", json={"answer": "The euro will go up."}).status_code == 200


@pytest.mark.parametrize("path", ["/", "/static/app.js", "/api/model"])
def test_browser_responses_are_not_cached(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
