"""Train a pretrained semantic encoder + temporal classifier, using validation only."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.sparse import csr_matrix, hstack
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.dummy import DummyClassifier

from flagger.model import make_pipeline
from flagger.semantic import ENCODER_ID, ENCODER_REVISION, SentenceEmbeddings, prepare_encoder
from flagger.text import normalize_text
from scripts.train import load_split, evaluate, save_artifact, render_report


def run(data_dir, output_dir, report_dir, offline=False):
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    train = load_split(data_dir, "train")
    validation = load_split(data_dir, "validation")
    statement_indices = [i for i, row in enumerate(validation) if row["source"] == "project_authored"]
    statement_validation = [validation[i] for i in statement_indices]
    if not statement_validation or set(row["label"] for row in statement_validation) != {0, 1}:
        raise ValueError("Validation needs statements from both classes")
    texts = [row["text"] for row in train]
    labels = np.array([row["label"] for row in train])
    weights = np.array([4.0 if row["source"] == "project_authored" else 1.0 for row in train])

    print("Preparing pinned MiniLM encoder (CPU; fixed pretrained weights)…", flush=True)
    encoding_started = perf_counter()
    encoder_dir = prepare_encoder(output_dir / "semantic_encoder", offline=offline)
    semantic = SentenceEmbeddings(str(encoder_dir.resolve())).fit(texts)
    embeddings = {}
    for name, rows in [("train", train), ("validation", validation)]:
        print(f"Encoding {len(rows):,} {name} sentences…", flush=True)
        embeddings[name] = semantic.transform([row["text"] for row in rows])
    encoding_seconds = perf_counter() - encoding_started

    baseline = make_pipeline("word_logistic")
    baseline_started = perf_counter()
    baseline.fit(texts, labels, classifier__sample_weight=weights)
    baseline_seconds = perf_counter() - baseline_started
    word = baseline.named_steps["features"]
    lexical = {name: word.transform([row["text"] for row in rows]) for name, rows in [("train", train), ("validation", validation)]}
    validation_labels = [row["label"] for row in statement_validation]

    # Fixed grid; only existing validation labels choose settings. Neither
    # original test set nor the paraphrase diagnostic participates in selection.
    trials = []
    fitted = {}
    for semantic_weight in [0.5, 1.0, 2.0, 4.0, "semantic_only"]:
        matrices = {
            name: csr_matrix(embeddings[name]) if semantic_weight == "semantic_only" else
            hstack([lexical[name], csr_matrix(embeddings[name] * semantic_weight)], format="csr")
            for name in embeddings
        }
        for regularization in [1.0, 4.0, 16.0]:
            head = LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            head_started = perf_counter()
            head.fit(matrices["train"], labels, sample_weight=weights)
            head_seconds = perf_counter() - head_started
            prediction = head.predict(matrices["validation"][statement_indices])
            score = float(f1_score(validation_labels, prediction, average="macro"))
            key = f"semantic={semantic_weight},C={regularization}"
            trial = {"key": key, "semantic_weight": semantic_weight, "C": regularization, "validation_macro_f1": score, "head_train_seconds": round(head_seconds, 4)}
            trials.append(trial)
            fitted[key] = head
            print(f"{key}: validation statement macro-F1={score:.3f}", flush=True)

    selected_trial = max(trials, key=lambda trial: (trial["validation_macro_f1"], -trial["C"]))
    best_by_kind = {}
    pipelines = {"word_logistic": baseline}
    for kind, only_semantic in [("hybrid_logistic", False), ("semantic_logistic", True)]:
        matching = [trial for trial in trials if (trial["semantic_weight"] == "semantic_only") == only_semantic]
        trial = max(matching, key=lambda item: (item["validation_macro_f1"], -item["C"]))
        best_by_kind[kind] = trial
        transformers = [("semantic", semantic)] if only_semantic else [("word", word), ("semantic", semantic)]
        transformer_weights = {"semantic": 1.0 if only_semantic else trial["semantic_weight"]}
        features = FeatureUnion(transformers, transformer_weights=transformer_weights)
        pipelines[kind] = Pipeline([("features", features), ("classifier", fitted[trial["key"]])])

    selected = "semantic_logistic" if selected_trial["semantic_weight"] == "semantic_only" else "hybrid_logistic"
    model = pipelines[selected]
    candidates = {}
    for kind, pipeline in pipelines.items():
        seconds = baseline_seconds if kind == "word_logistic" else encoding_seconds + best_by_kind[kind]["head_train_seconds"]
        candidates[kind] = {
            "train_seconds": round(seconds, 4),
            "validation_statements": evaluate(pipeline, statement_validation),
            "validation_all": evaluate(pipeline, validation),
        }

    # Evaluation begins after selection. Preserve and report every error.
    tests = {"questions": load_split(data_dir, "test_questions"), "statements": load_split(data_dir, "test_statements")}
    challenge_path = ROOT / "data/paraphrase_check.tsv"
    with challenge_path.open(encoding="utf-8", newline="") as handle:
        challenge = list(csv.DictReader(handle, delimiter="\t"))
    for row in challenge:
        row["label"] = int(row["label"])
    known = {normalize_text(row["text"]) for row in train + validation + [row for rows in tests.values() for row in rows]}
    if known & {normalize_text(row["text"]) for row in challenge}:
        raise ValueError("Paraphrase diagnostic overlaps an existing dataset split")
    comparisons = {
        kind: {name: evaluate(pipelines[kind], rows) for name, rows in tests.items()}
        for kind in ["word_logistic", selected]
    }
    paraphrases = {kind: evaluate(pipelines[kind], challenge) for kind in ["word_logistic", selected]}
    test_metrics = comparisons[selected]
    dummy = DummyClassifier(strategy="most_frequent").fit(texts, labels)
    dummy_metrics = {name: evaluate(dummy, rows) for name, rows in tests.items()}
    errors = []
    for name, rows in {**tests, "paraphrase_diagnostic": challenge}.items():
        probabilities = model.predict_proba([row["text"] for row in rows])[:, 1]
        for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
            if int(probability >= 0.5) != row["label"]:
                errors.append({"evaluation_set": name, "id": row.get("id", f"paraphrase_{index:03}"), "text": row["text"], "gold_label": row["label"], "prediction": int(probability >= 0.5), "volatility_probability": round(float(probability), 6)})
    with (report_dir / "errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["evaluation_set", "id", "text", "gold_label", "prediction", "volatility_probability"])
        writer.writeheader()
        writer.writerows(errors)

    names = model.named_steps["features"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    indices = sorted([i for i, name in enumerate(names) if name.startswith("word__")], key=lambda i: float(coefficients[i]))
    top_features = {
        "stable": [{"feature": names[i].removeprefix("word__"), "weight": round(float(coefficients[i]), 4)} for i in indices[:20]],
        "volatile": [{"feature": names[i].removeprefix("word__"), "weight": round(float(coefficients[i]), 4)} for i in indices[-20:][::-1]],
        "note": "These are only the lexical branch. Contextual embedding dimensions are not individual words.",
    }
    (report_dir / "top_features.json").write_text(json.dumps(top_features, indent=2) + "\n")
    benchmark = [row["text"] for row in tests["statements"][:20]]
    model.predict_proba(benchmark)
    benchmark_started = perf_counter()
    for _ in range(5):
        model.predict_proba(benchmark)
    batch_ms = (perf_counter() - benchmark_started) * 1000 / 5
    manifest = json.loads((data_dir / "manifest.json").read_text())
    metadata = {
        "name": "AI Answer Staleness Flagger", "model_kind": selected,
        "algorithm": "MiniLM sentence embeddings + " + ("TF-IDF + " if selected == "hybrid_logistic" else "") + "Logistic Regression",
        "labels": {"0": "STABLE", "1": "TIME-SENSITIVE"}, "language": "English", "threshold": 0.5,
        "training_rows": len(train), "feature_count": len(names), "embedding_dimensions": 384,
        "train_seconds": round(encoding_seconds + selected_trial["head_train_seconds"], 4),
        "encoding_seconds": round(encoding_seconds, 4), "head_train_seconds": selected_trial["head_train_seconds"],
        "batch_20_ms": round(batch_ms, 3), "statement_test": {key: value for key, value in test_metrics["statements"].items() if key != "classification_report"},
        "question_test": {key: value for key, value in test_metrics["questions"].items() if key != "classification_report"},
        "semantic_encoder": {"model": ENCODER_ID, "revision": ENCODER_REVISION, "directory": "semantic_encoder", "frozen": True, "max_tokens": 256},
        "selected_parameters": selected_trial,
        "annotation_note": manifest["annotation_note"],
        "probability_note": "Uncalibrated temporal-volatility scores, not factual correctness or actual staleness.",
        "trained_at": datetime.now(timezone.utc).isoformat(), "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "data_sha256": hashlib.sha256((data_dir / "train.csv").read_bytes()).hexdigest(),
        "offline_inference": True, "fine_tuning": False,
        "selection_note": "Semantic settings selected by macro-F1 on existing validation statements only; test and paraphrase labels excluded",
    }
    artifact_path = output_dir / "staleness.joblib"
    save_artifact({"format_version": 1, "pipeline": model, "metadata": metadata}, artifact_path)
    metadata["artifact_bytes"] = artifact_path.stat().st_size
    metadata["encoder_bytes"] = sum(path.stat().st_size for path in encoder_dir.rglob("*") if path.is_file())
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = {
        "selected_model": selected, "validation_selected_model": selected,
        "selection_metric": metadata["selection_note"], "candidates": candidates,
        "semantic_trials": trials, "test": test_metrics, "majority_baseline": dummy_metrics,
        "baseline_comparison": comparisons, "paraphrase_diagnostic": paraphrases,
        "paraphrase_note": "60 assistant-authored, rubric-labeled statements grouped into 20 paraphrase families; not independently human-reviewed, not used for model selection",
        "data": manifest, "metadata": metadata,
    }
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    render_report(report, report_dir)
    with (report_dir / "evaluation.md").open("a") as handle:
        handle.write("\n## Semantic comparison\n\nA fixed, pretrained MiniLM encoder supplies contextual sentence vectors. A supervised classifier combines these with lexical cues that help preserve tense and historical anchoring. The encoder is not fine-tuned on this project. The one-second lexical fitting time did not include any semantic pretraining.\n\n| Model | Statement accuracy | Statement macro-F1 | Paraphrase accuracy |\n|---|---:|---:|---:|\n")
        for kind in ["word_logistic", selected]:
            handle.write(f"| {kind} | {comparisons[kind]['statements']['accuracy']:.3f} | {comparisons[kind]['statements']['macro_f1']:.3f} | {paraphrases[kind]['accuracy']:.3f} |\n")
        handle.write("\n" + report["paraphrase_note"] + ". These figures evaluate assigned temporal labels, not factual truth. Inspect all remaining failures in `errors.csv`.\n")
    print(f"Selected {selected}: {selected_trial}", flush=True)
    print(f"Representation preparation: {encoding_seconds:.2f}s; selected head: {selected_trial['head_train_seconds']:.2f}s; full comparison run: {perf_counter() - started:.2f}s", flush=True)
    print(f"Statement accuracy: {test_metrics['statements']['accuracy']:.3f}; paraphrase diagnostic: {paraphrases[selected]['accuracy']:.3f}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--offline", action="store_true", help="Use only an already cached pretrained checkpoint")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.report_dir, args.offline)
