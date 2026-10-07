import argparse
import json
import sys

from flagger.predict import StalenessClassifier


def main():
    parser = argparse.ArgumentParser(description="Predict temporal volatility, not truth.")
    parser.add_argument("answer", nargs="?", help="Answer text; otherwise reads stdin")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", help="Print structured predictions")
    args = parser.parse_args()
    answer = args.answer if args.answer is not None else sys.stdin.read()
    if not answer.strip():
        parser.error("An answer is required")
    try:
        result = StalenessClassifier().analyze(answer, args.threshold)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for claim in result["claims"]:
            icon = "🔴" if claim["label"] else "🟢"
            print(f"{icon} {claim['label']} {claim['label_name']} | {claim['text']}")
        print("\nLabels describe change over time, not factual correctness.")


if __name__ == "__main__":
    main()
