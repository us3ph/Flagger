"""One-time public download; all later training and inference can be offline."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import GroupShuffleSplit
from flagger.text import normalize_text, split_claims

REPO_REVISION = "88f9759fbf66abfd45c454fe4d9d2db20cd6a5da"
HF_REVISION = "0a114a0126272e991a368dddcbe3092bd72dbd16"
FIELDS = ["id", "text", "label", "group", "source", "kind", "annotation_status", "source_question", "source_url"]


def read_csv(path, delimiter=","):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def group_for(text):
    return "eg_" + hashlib.sha256(normalize_text(text).encode()).hexdigest()[:16]


def curated(path):
    rows = read_csv(path, "\t")
    return [dict(
        id=f"{path.stem}_{i:03}", text=row["text"], label=int(row["label"]),
        group="curated_" + row["group"], source="project_authored", kind="statement",
        annotation_status="assistant_authored_rubric_label_needs_human_review",
        source_question="", source_url="",
    ) for i, row in enumerate(rows)]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare(offline=False):
    raw = ROOT / "data/raw"
    output = ROOT / "data/processed"
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    upstream = {}
    downloads = []
    for split in ["train", "train_synth", "test"]:
        url = f"https://raw.githubusercontent.com/s-nlp/Evergreen/{REPO_REVISION}/datasets/{split}.csv"
        path = raw / f"evergreen_{split}.csv"
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"Missing {path}. Run once without --offline to download.")
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        upstream[split] = read_csv(path)
        downloads.append({"split": split, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(upstream[split])})

    # Assert the semantics using upstream field names and unmistakable anchors.
    anchors = {row["English"]: int(row["is_evergreen"]) for row in upstream["train"]}
    assert anchors["In what year was Facebook founded?"] == 1
    synth_anchors = {row["English"]: int(row["is_evergreen"]) for row in upstream["train_synth"]}
    assert synth_anchors["What is the current price of Bitcoin?"] == 0

    official_test = []
    for i, row in enumerate(upstream["test"]):
        text = row["English"].strip()
        if text:
            official_test.append(dict(id=f"eg_test_{i}", text=text, label=1 - int(row["is_evergreen"]),
                group=group_for(text), source="evergreen_test", kind="question", annotation_status="upstream_label_inverted",
                source_question=text, source_url=downloads[2]["url"]))
    authored_test = curated(ROOT / "data/curated_test.tsv")
    reserved = {normalize_text(row["text"]) for row in official_test + authored_test}
    questions = []
    answers = []
    for split in ["train", "train_synth"]:
        url = next(item["url"] for item in downloads if item["split"] == split)
        for i, row in enumerate(upstream[split]):
            text = row["English"].strip()
            if not text or normalize_text(text) in reserved:
                continue
            label = 1 - int(row["is_evergreen"])
            assert label in {0, 1}
            common = dict(group=group_for(text), source=f"evergreen_{split}", source_question=text, source_url=url)
            questions.append(dict(id=f"eg_{split}_{i}", text=text, label=label, kind="question",
                                  annotation_status="upstream_label_inverted", **common))
            answer = row.get("en_Answer", "").strip()
            # Question labels do not automatically label historical subclaims.
            # Keep short, single-sentence answers; filter retrospective volatile
            # answers. These are still explicitly WEAK labels, not gold data.
            if not 15 <= len(answer) <= 200 or "\\n" in answer or len(split_claims(answer)) != 1:
                continue
            if label == 1 and re.search(r"\b(?:was|were|had|in\s+\d{4}|from\s+\d{4}|during\s+\d{4})\b", answer, re.I):
                continue
            if normalize_text(answer) in reserved:
                continue
            answers.append(dict(id=f"eg_answer_{i}", text=answer, label=label, kind="statement",
                                annotation_status="weak_question_label_not_human_claim_annotation", **common))

    # Bound adaptation to 150 examples of each label, in original source order.
    selected_answers = []
    for label in [0, 1]:
        selected_answers.extend([row for row in answers if row["label"] == label][:150])
    rows = questions + selected_answers + curated(ROOT / "data/curated_train.tsv")

    # Merge exact duplicate groups before splitting; conflicting labels are
    # excluded, never silently resolved using a majority or a test label.
    by_text = {}
    conflicting = set()
    for row in rows:
        key = normalize_text(row["text"])
        if key in reserved:
            continue
        if key in by_text:
            if row["label"] != by_text[key]["label"]:
                conflicting.add(key)
        else:
            by_text[key] = row
    rows = [row for key, row in by_text.items() if key not in conflicting]

    # A source question and its answer always stay in the same split. Related
    # authored examples share a family group, keeping paraphrases together.
    train_indices, val_indices = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(rows, groups=[r["group"] for r in rows]))
    train = [rows[int(i)] for i in train_indices]
    validation = [rows[int(i)] for i in val_indices]
    # User-provided examples are regression examples, never evaluation evidence.
    required_groups = {"curated_python", "curated_earth", "curated_generic_price", "curated_visa", "curated_ceo"}
    train += [r for r in validation if r["group"] in required_groups]
    validation = [r for r in validation if r["group"] not in required_groups]

    datasets = {"train": train, "validation": validation, "test_questions": official_test, "test_statements": authored_test}
    audit = audit_splits(datasets)
    for split, records in datasets.items():
        write_csv(output / f"{split}.csv", records)
    manifest = {
        "seed": 42, "upstream_hf_dataset": "s-nlp/EverGreen-Multilingual", "hf_revision": HF_REVISION,
        "github_revision": REPO_REVISION, "downloads": downloads,
        "mapping": {"upstream_is_evergreen_1": 0, "upstream_is_evergreen_0": 1},
        "conflicting_training_texts_removed": len(conflicting), "audit": audit,
        "split_counts": {name: {"total": len(records), "stable": sum(r["label"] == 0 for r in records),
                                    "volatile": sum(r["label"] == 1 for r in records),
                                    "curated_statements": sum(r["source"] == "project_authored" for r in records)}
                         for name, records in datasets.items()},
        "dataset_license": "No explicit dataset license found in the Hugging Face card; raw upstream files stay in gitignored data/raw.",
        "annotation_note": "Project-authored statements were labeled by the coding assistant with the rubric. They are not independent human annotations. Short public answers inherit weak question labels.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["split_counts"], indent=2))
    return manifest


def audit_splits(datasets):
    names = list(datasets)
    checks = []
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            left, right = datasets[name], datasets[other]
            text_overlap = {normalize_text(r["text"]) for r in left} & {normalize_text(r["text"]) for r in right}
            group_overlap = {r["group"] for r in left} & {r["group"] for r in right}
            if text_overlap or group_overlap:
                raise ValueError(f"Split leakage between {name} and {other}: {len(text_overlap)} texts, {len(group_overlap)} groups")
            checks.append({"splits": [name, other], "exact_text_overlap": 0, "group_overlap": 0})
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Require cached source CSV files")
    args = parser.parse_args()
    prepare(args.offline)
