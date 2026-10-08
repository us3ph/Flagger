"""Local pretrained sentence representations for semantic classification."""

from functools import lru_cache
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from flagger.text import normalize_text

ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_DIMENSIONS = 384


def prepare_encoder(destination: Path, offline: bool = False) -> Path:
    """Copy a pinned checkpoint once; runtime never downloads model files."""
    manifest = destination / "flagger_encoder.json"
    expected = {"model": ENCODER_ID, "revision": ENCODER_REVISION}
    if manifest.is_file() and json.loads(manifest.read_text()) == expected:
        if (destination / "model.safetensors").is_file():
            return destination
    from huggingface_hub import snapshot_download
    source = snapshot_download(
        repo_id=ENCODER_ID, revision=ENCODER_REVISION, local_files_only=offline,
        allow_patterns=["*.json", "model.safetensors", "vocab.txt", "1_Pooling/*", "README.md"],
    )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    manifest.write_text(json.dumps(expected, indent=2) + "\n")
    return destination


@lru_cache(maxsize=2)
def get_encoder(model_dir: str):
    if not (Path(model_dir) / "model.safetensors").is_file():
        raise FileNotFoundError("Semantic encoder missing. Run: python scripts/train_semantic.py")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    encoder = SentenceTransformer(model_dir, device="cpu", local_files_only=True)
    encoder.max_seq_length = 256
    return encoder


class SentenceEmbeddings(TransformerMixin, BaseEstimator):
    """Frozen contextual transformer features, with a separately trained head."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def fit(self, texts, y=None):
        self.n_features_out_ = EMBEDDING_DIMENSIONS
        return self

    def transform(self, texts):
        if not len(texts):
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        return get_encoder(self.model_dir).encode(
            [normalize_text(text) for text in texts],
            batch_size=32, normalize_embeddings=True, show_progress_bar=False,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array([f"embedding_{index}" for index in range(EMBEDDING_DIMENSIONS)])
