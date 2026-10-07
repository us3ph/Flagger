import pytest
from fastapi.testclient import TestClient

from flagger.app import app


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
