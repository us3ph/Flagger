import joblib
import pytest

from scripts.train import save_artifact


def test_failed_artifact_write_keeps_previous_model(tmp_path, monkeypatch):
    path = tmp_path / "staleness.joblib"
    save_artifact({"revision": "first"}, path)

    def interrupted_write(artifact, temporary_path, **kwargs):
        temporary_path.write_bytes(b"partial artifact")
        assert joblib.load(path) == {"revision": "first"}
        raise OSError("Interrupted training output")

    monkeypatch.setattr(joblib, "dump", interrupted_write)
    with pytest.raises(OSError, match="Interrupted"):
        save_artifact({"revision": "second"}, path)
    assert joblib.load(path) == {"revision": "first"}
    assert list(tmp_path.iterdir()) == [path]
