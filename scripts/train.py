"""Train and compare small classifiers on CPU. No pretrained model fine-tuning."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from flagger.model import make_pipeline


def load_split(directory, name):
    path = directory / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError("Prepared data missing. Run: python scripts/prepare_data.py")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        if row["label"] not in {0, 1} or not row["text"].strip():
            raise ValueError(f"Invalid example in {path}")
    return rows


def evaluate(model, rows):
    labels = [row["label"] for row in rows]
    predictions = model.predict([row["text"] for row in rows])
    report = classification_report(labels, predictions, labels=[0, 1], target_names=["STABLE", "TIME-SENSITIVE"], output_dict=True, zero_division=0)
    return {
        "n": len(rows), "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "volatile_f1": report["TIME-SENSITIVE"]["f1-score"],
        "volatile_precision": report["TIME-SENSITIVE"]["precision"],
        "volatile_recall": report["TIME-SENSITIVE"]["recall"],
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": report,
    }


def run(data_dir, output_dir, report_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    train = load_split(data_dir, "train")
    validation = load_split(data_dir, "validation")
    statement_validation = [r for r in validation if r["source"] == "project_authored"]
    if not statement_validation or set(r["label"] for r in statement_validation) != {0, 1}:
        raise ValueError("Validation needs curated statements from both classes")
    x_train = [r["text"] for r in train]
    y_train = np.array([r["label"] for r in train])
    # Fixed before evaluation: give target-format statements more influence
    # than auxiliary questions. Weak answer labels get ordinary weight.
    weights = np.array([4.0 if r["source"] == "project_authored" else 1.0 for r in train])
    candidates = {}
    pipelines = {}
    for kind in ["word_logistic", "word_char_logistic", "word_naive_bayes"]:
        model = make_pipeline(kind)
        started = perf_counter()
        model.fit(x_train, y_train, classifier__sample_weight=weights)
        train_seconds = perf_counter() - started
        pipelines[kind] = model
        candidates[kind] = {
            "train_seconds": round(train_seconds, 4),
            "validation_statements": evaluate(model, statement_validation),
            "validation_all": evaluate(model, validation),
        }
        print(f"{kind}: {train_seconds:.2f}s, statement validation macro-F1={candidates[kind]['validation_statements']['macro_f1']:.3f}")
    # Select only with validation labels. Prefer Logistic Regression on ties.
    selected = max(candidates, key=lambda k: (candidates[k]["validation_statements"]["macro_f1"], k != "word_naive_bayes", k == "word_logistic"))
    model = pipelines[selected]
    tests = {"questions": load_split(data_dir, "test_questions"), "statements": load_split(data_dir, "test_statements")}
    test_metrics = {name: evaluate(model, rows) for name, rows in tests.items()}
    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    dummy_metrics = {name: evaluate(dummy, rows) for name, rows in tests.items()}

    errors = []
    for name, rows in tests.items():
        probabilities = model.predict_proba([r["text"] for r in rows])[:, 1]
        for row, probability in zip(rows, probabilities, strict=True):
            prediction = int(probability >= 0.5)
            if prediction != row["label"]:
                errors.append({"evaluation_set": name, "id": row["id"], "text": row["text"], "gold_label": row["label"], "prediction": prediction, "volatility_probability": round(float(probability), 6)})
    with (report_dir / "errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["evaluation_set", "id", "text", "gold_label", "prediction", "volatility_probability"])
        writer.writeheader()
        writer.writerows(errors)

    features = model.named_steps["features"].get_feature_names_out()
    classifier = model.named_steps["classifier"]
    coefficients = classifier.coef_[0] if hasattr(classifier, "coef_") else classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    word_indices = [i for i, name in enumerate(features) if "char__" not in name]
    sorted_indices = sorted(word_indices, key=lambda i: float(coefficients[i]))
    top_features = {
        "stable": [{"feature": features[i].removeprefix("word__"), "weight": round(float(coefficients[i]), 4)} for i in sorted_indices[:20]],
        "volatile": [{"feature": features[i].removeprefix("word__"), "weight": round(float(coefficients[i]), 4)} for i in sorted_indices[-20:][::-1]],
    }
    (report_dir / "top_features.json").write_text(json.dumps(top_features, indent=2) + "\n")

    benchmark_texts = [r["text"] for r in tests["statements"][:20]]
    model.predict_proba(benchmark_texts)
    started = perf_counter()
    for _ in range(30):
        model.predict_proba(benchmark_texts)
    batch_ms = (perf_counter() - started) * 1000 / 30
    manifest = json.loads((data_dir / "manifest.json").read_text())
    metadata = {
        "name": "AI Answer Staleness Flagger", "model_kind": selected,
        "algorithm": "TF-IDF + " + ("Multinomial Naive Bayes" if selected == "word_naive_bayes" else "Logistic Regression"),
        "labels": {"0": "STABLE", "1": "TIME-SENSITIVE"}, "language": "English",
        "threshold": 0.5, "feature_count": len(features), "training_rows": len(train),
        "train_seconds": candidates[selected]["train_seconds"], "batch_20_ms": round(batch_ms, 3),
        "statement_test": {k: v for k, v in test_metrics["statements"].items() if k != "classification_report"},
        "question_test": {k: v for k, v in test_metrics["questions"].items() if k != "classification_report"},
        "annotation_note": manifest["annotation_note"],
        "probability_note": "Scores are classifier estimates, not calibrated probabilities of correctness or actual staleness.",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "data_sha256": hashlib.sha256((data_dir / "train.csv").read_bytes()).hexdigest(),
        "offline_inference": True, "fine_tuning": False,
    }
    artifact_path = output_dir / "staleness.joblib"
    joblib.dump({"format_version": 1, "pipeline": model, "metadata": metadata}, artifact_path, compress=3)
    metadata["artifact_bytes"] = artifact_path.stat().st_size
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = {"selected_model": selected, "selection_metric": "macro-F1 on project-authored validation statements only",
              "candidates": candidates, "test": test_metrics, "majority_baseline": dummy_metrics,
              "data": manifest, "metadata": metadata}
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    render_report(report, report_dir)
    print(f"Selected: {selected}. Artifact: {artifact_path.stat().st_size / 1024:.1f} KiB")
    for name, values in test_metrics.items():
        print(f"Held-out {name}: n={values['n']}, accuracy={values['accuracy']:.3f}, macro-F1={values['macro_f1']:.3f}")
    return report


def render_report(report, directory):
    lines = ["# Evaluation report", "", "Selection uses validation statements only. Both test sets remain outside training and selection. The saved classifier is the evaluated training-only model; validation is not folded into it.", "", "## Validation comparison", "", "| Model | Train seconds | Statement macro-F1 | All-data macro-F1 |", "|---|---:|---:|---:|"]
    for name, values in report["candidates"].items():
        lines.append(f"| {name} | {values['train_seconds']:.3f} | {values['validation_statements']['macro_f1']:.3f} | {values['validation_all']['macro_f1']:.3f} |")
    lines += ["", f"Selected: **{report['selected_model']}**.", "", "## Held-out evaluation", "", "| Set | N | Accuracy | Macro-F1 | Volatile precision | Volatile recall | Majority macro-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in report["test"].items():
        lines.append(f"| {name} | {values['n']} | {values['accuracy']:.3f} | {values['macro_f1']:.3f} | {values['volatile_precision']:.3f} | {values['volatile_recall']:.3f} | {report['majority_baseline'][name]['macro_f1']:.3f} |")
        lines += []
    for name, values in report["test"].items():
        matrix = values["confusion_matrix"]
        lines += ["", f"### {name.title()} confusion matrix", "", "Rows are gold labels; columns are predictions.", "", "| | Predicted stable | Predicted volatile |", "|---|---:|---:|", f"| Gold stable | {matrix[0][0]} | {matrix[0][1]} |", f"| Gold volatile | {matrix[1][0]} | {matrix[1][1]} |"]
    lines += ["", "## Interpretation", "", "The official test contains questions, so its score does not measure answer-sentence performance. The statement test is a small assistant-authored exercise labeled with the project rubric; it is not an independently human-annotated benchmark. Both question-to-statement transfer and annotation bias limit generalization. Short source answers are weakly labeled and may contain noise. Exact text and source/family groups are disjoint across splits; semantic overlap between independently written examples can still exist.", "", "Inspect `errors.csv` for every test error and `top_features.json` for learned lexical associations. The system labels volatility, including potentially false claims; it never verifies correctness.", "", "Probabilities are not calibrated. Timing is a local CPU measurement and excludes HTTP and startup overhead.", ""]
    (directory / "evaluation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.report_dir)
