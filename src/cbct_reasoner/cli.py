from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cbct_reasoner.data import discover_cases, write_manifest
from cbct_reasoner.metrics import evaluate_pairs
from cbct_reasoner.model import RetrievalReportModel
from cbct_reasoner.reporting import read_challenge_output, write_challenge_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbct-reasoner",
        description="ToothFairy4 CBCT-to-report baseline utilities",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="validate a dataset and write a case manifest")
    index.add_argument("--data", type=Path, required=True)
    index.add_argument("--output", type=Path, default=Path("artifacts/manifest.jsonl"))

    train = commands.add_parser("train", help="fit the retrieval baseline")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("artifacts/retrieval_model.npz"))

    predict = commands.add_parser("predict", help="predict one local NIfTI/MHA case")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument(
        "--output", type=Path, default=Path("predictions/diagnostic-imaging-report.json")
    )
    predict.add_argument("--show-provenance", action="store_true")

    validate = commands.add_parser("validate-output", help="validate challenge output JSON")
    validate.add_argument("path", type=Path)

    evaluate = commands.add_parser("evaluate", help="score a JSONL prediction/reference file")
    evaluate.add_argument("--pairs", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "index":
            records = discover_cases(args.data)
            write_manifest(records, args.output)
            print(json.dumps({"cases": len(records), "manifest": str(args.output)}))
        elif args.command == "train":
            records = discover_cases(args.data)
            model = RetrievalReportModel.fit(records)
            model.save(args.output)
            print(json.dumps({"cases": len(records), "model": str(args.output)}))
        elif args.command == "predict":
            prediction = RetrievalReportModel.load(args.model).predict_path(args.input)
            write_challenge_output(prediction.report, args.output)
            result: dict[str, object] = {"output": str(args.output)}
            if args.show_provenance:
                result.update(
                    source_case_id=prediction.source_case_id,
                    distance=prediction.distance,
                )
            print(json.dumps(result))
        elif args.command == "validate-output":
            report = read_challenge_output(args.path)
            print(json.dumps({"valid": True, "characters": len(report)}))
        elif args.command == "evaluate":
            metrics = _evaluate_jsonl(args.pairs)
            rendered = json.dumps(metrics, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
        else:  # pragma: no cover - argparse guarantees a registered command
            raise AssertionError(f"Unhandled command: {args.command}")
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _evaluate_jsonl(path: Path) -> dict[str, float | int]:
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            try:
                prediction, reference = row["prediction"], row["reference"]
            except KeyError as error:
                raise ValueError(f"line {line_number} lacks {error.args[0]!r}") from error
            if not isinstance(prediction, str) or not isinstance(reference, str):
                raise ValueError(f"line {line_number} prediction/reference must be strings")
            pairs.append((prediction, reference))
    return evaluate_pairs(pairs)


if __name__ == "__main__":
    raise SystemExit(main())
