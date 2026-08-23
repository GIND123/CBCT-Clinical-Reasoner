"""Assemble every produced number into one results file and a readable summary.

Run after the pipeline finishes. The output feeds the Hugging Face model card and
is the artifact to cite when comparing runs, so it records provenance (commit,
config, dataset counts) alongside the metrics rather than the metrics alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_reasoner.config import ExperimentConfig, Paths, default_paths  # noqa: E402


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def collect(paths: Paths, config: ExperimentConfig) -> dict[str, Any]:
    manifest = paths.manifest
    cases = sum(1 for _ in manifest.open(encoding="utf-8")) if manifest.is_file() else 0

    bank = _read(paths.prototypes) or {}
    history = _read(paths.artifacts / "train_history.json") or []
    calibration = _read(paths.artifacts / "calibration.json") or {}
    evaluation = _read(paths.artifacts / "evaluation.json") or {}
    ablation = _read(paths.artifacts / "ablation.json") or {}

    fold_scores = [item.get("best_score") for item in history if item.get("best_score") is not None]

    return {
        "provenance": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "config": config.to_dict(),
        },
        "dataset": {
            "cases": cases,
            "prototypes": len(bank.get("prototypes", [])),
            "assign_threshold": bank.get("assign_threshold"),
            "tooth_aware": bank.get("tooth_aware"),
        },
        "training": {
            "folds": len(history),
            "fold_val_map": fold_scores,
            "mean_val_map": sum(fold_scores) / len(fold_scores) if fold_scores else None,
        },
        "calibration": {
            "objective": calibration.get("objective"),
            "baseline_fixed_threshold": calibration.get("baseline"),
            "rounds": calibration.get("rounds"),
        },
        "evaluation": evaluation.get("surrogate"),
        "evaluation_radfact_lite": evaluation.get("radfact_lite"),
        "ablation": ablation,
        "caveats": [
            "Clinical scores are the repository's offline RadFact surrogate unless "
            "'evaluation_radfact_lite' is populated. The surrogate ranks decoder "
            "variants; it is not the challenge metric.",
            "BLEU-4 and METEOR are exact reimplementations of the grader's local "
            "implementations and match NLTK to machine precision.",
            "Scores are out-of-fold on the public training release, not the hidden "
            "50-case test set, and the folds are stratified rather than "
            "leave-one-centre-out, so they measure in-domain performance.",
        ],
    }


def render(results: dict[str, Any]) -> str:
    lines = ["# Results", ""]
    dataset = results["dataset"]
    lines += [
        f"- cases: **{dataset['cases']}**",
        f"- prototypes: **{dataset['prototypes']}** "
        f"(tooth-aware={dataset['tooth_aware']}, assign threshold={dataset['assign_threshold']})",
    ]
    training = results["training"]
    if training["mean_val_map"] is not None:
        folds = ", ".join(f"{value:.4f}" for value in training["fold_val_map"])
        lines += [f"- out-of-fold mAP: **{training['mean_val_map']:.4f}** (folds: {folds})"]

    evaluation = results.get("evaluation")
    if evaluation:
        lines += [
            "",
            "## Out-of-fold score",
            "",
            "| metric | value |",
            "|---|---|",
            f"| final (0.8 clinical + 0.2 captioning) | **{evaluation['final']:.4f}** |",
            f"| clinical F1 (surrogate) | {evaluation['clinical']:.4f} |",
            f"| logical precision | {evaluation['logical_precision']:.4f} |",
            f"| logical recall | {evaluation['logical_recall']:.4f} |",
            f"| BLEU-4 (grader-exact) | {evaluation['bleu_4']:.4f} |",
            f"| METEOR (grader-exact) | {evaluation['meteor']:.4f} |",
        ]

    ablation = results.get("ablation") or {}
    if ablation:
        lines += [
            "",
            "## Ablation",
            "",
            "| variant | final | clinical | BLEU-4 | METEOR |",
            "|---|---|---|---|---|",
        ]
        for name, values in ablation.items():
            lines.append(
                f"| {name} | {values['final']:.4f} | {values['clinical']:.4f} | "
                f"{values['bleu_4']:.4f} | {values['meteor']:.4f} |"
            )

    lines += ["", "## Caveats", ""] + [f"- {item}" for item in results["caveats"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/toothfairy4.json"))
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()

    paths = default_paths()
    if args.work:
        paths = paths.with_root(args.work)
    config = ExperimentConfig.load(args.config if args.config.is_file() else None)

    results = collect(paths, config)
    (paths.artifacts / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    summary = render(results)
    (paths.artifacts / "RESULTS.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
