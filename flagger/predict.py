"""Offline, batched inference from a trusted local model artifact."""

from pathlib import Path
from time import perf_counter
import joblib
import numpy as np

from flagger.text import split_claims

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "staleness.joblib"
LABELS = {0: "STABLE", 1: "TIME-SENSITIVE"}


class StalenessClassifier:
    def __init__(self, model_path: Path = DEFAULT_MODEL):
        if not model_path.is_file():
            raise FileNotFoundError("Model missing. Run: python scripts/train.py")
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.metadata = artifact["metadata"]
        self.version = artifact["format_version"]
        if self.version != 1 or list(self.pipeline.classes_) != [0, 1]:
            raise ValueError("Unsupported model artifact or label mapping")

    def predict_claims(self, texts: list[str], threshold: float = 0.5) -> list[dict]:
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
        if not texts:
            return []
        probabilities = self.pipeline.predict_proba(texts)[:, 1]
        return [
            {
                "text": text,
                "label": int(probability >= threshold),
                "label_name": LABELS[int(probability >= threshold)],
                "volatility_probability": round(float(probability), 6),
            }
            for text, probability in zip(texts, probabilities, strict=True)
        ]

    def analyze(self, answer: str, threshold: float = 0.5, max_claims: int = 150) -> dict:
        started = perf_counter()
        spans = split_claims(answer)
        if len(spans) > max_claims:
            raise ValueError(f"Please use at most {max_claims} sentences per answer")
        predictions = self.predict_claims([span.text for span in spans], threshold)
        for prediction, span in zip(predictions, spans, strict=True):
            prediction.update(start=span.start, end=span.end)
        volatile = sum(item["label"] for item in predictions)
        return {
            "claims": predictions,
            "summary": {"total": len(predictions), "stable": len(predictions) - volatile, "volatile": volatile},
            "threshold": threshold,
            "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            "model": self.metadata["model_kind"],
            "scope": "Temporal volatility only. Labels do not indicate truth, falsehood, or actual staleness.",
        }
