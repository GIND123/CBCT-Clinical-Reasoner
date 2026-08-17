"""Fit the low-capacity model and compare it head-to-head with the neural run.

Same folds, same label space, same decoder - so the only thing that differs is
the predictor. Writes ``work/oof_shallow.npz`` in the format the calibrator
consumes, so either set of out-of-fold probabilities can be fed to
``cbct-reasoner calibrate``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_reasoner.config import ExperimentConfig, default_paths  # noqa: E402
from cbct_reasoner.data.splits import SplitPlan  # noqa: E402
from cbct_reasoner.models.shallow import (  # noqa: E402
    ShallowConfig,
    build_features,
    fit_out_of_fold,
)
from cbct_reasoner.prototypes import PrototypeBank, load_labels  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_encoder import auc  # noqa: E402


def report(
    name: str, probabilities: np.ndarray, labels: np.ndarray, prevalence: np.ndarray
) -> dict:
    rows = [
        (column, prevalence[column], int(labels[:, column].sum()), value)
        for column in range(labels.shape[1])
        if (value := auc(probabilities[:, column].astype(np.float64), labels[:, column]))
        is not None
    ]
    aucs = np.asarray([r[3] for r in rows])
    weights = np.asarray([r[1] for r in rows])
    supports = np.asarray([r[2] for r in rows])
    strong = supports >= 12
    summary = {
        "mean_auc": float(aucs.mean()),
        "prevalence_weighted_auc": float((aucs * weights).sum() / weights.sum()),
        "mean_auc_support_12plus": float(aucs[strong].mean()) if strong.any() else float("nan"),
        "n_support_12plus": int(strong.sum()),
    }
    print(
        f"{name:<12} mean AUC {summary['mean_auc']:.4f} | "
        f"prevalence-weighted {summary['prevalence_weighted_auc']:.4f} | "
        f"support>=12 {summary['mean_auc_support_12plus']:.4f} "
        f"(n={summary['n_support_12plus']})"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/toothfairy4.json"))
    parser.add_argument("--min-support", type=int, default=12)
    parser.add_argument("--C", dest="inverse_regularization", type=float, default=0.05)
    args = parser.parse_args()

    paths = default_paths()
    ExperimentConfig.load(args.config if args.config.is_file() else None)
    settings = ShallowConfig(
        min_support=args.min_support, inverse_regularization=args.inverse_regularization
    )

    bank = PrototypeBank.load(paths.prototypes)
    case_ids, labels = load_labels(paths.labels)
    plan = SplitPlan.load(paths.folds)

    feature_path = paths.work / "shallow_features.npy"
    if feature_path.is_file() and np.load(feature_path, mmap_mode="r").shape[0] == len(case_ids):
        features = np.load(feature_path)
        print(f"reusing cached features {features.shape}")
    else:
        started = time.time()
        features = build_features(paths.cache, case_ids, settings)
        np.save(feature_path, features)
        print(f"built features {features.shape} in {time.time() - started:.0f}s")

    started = time.time()
    probabilities, info = fit_out_of_fold(
        features, labels, case_ids, plan, bank.prevalence, settings
    )
    print(
        f"fitted {int(info['fitted_statements'])}/{int(info['total_statements'])} statements "
        f"in {time.time() - started:.0f}s"
    )

    np.savez_compressed(
        paths.work / "oof_shallow.npz",
        case_ids=np.asarray(case_ids, dtype=np.str_),
        probabilities=probabilities.astype(np.float32),
        covered=np.ones(len(case_ids), dtype=bool),
    )

    print("\nA prior-only model scores exactly 0.5000 by construction.\n")
    summaries = {"shallow": report("shallow", probabilities, labels, bank.prevalence)}
    if paths.oof.is_file():
        with np.load(paths.oof, allow_pickle=False) as archive:
            neural_ids = [str(v) for v in archive["case_ids"].tolist()]
            neural = archive["probabilities"].astype(np.float64)
        if neural_ids == list(case_ids):
            summaries["neural"] = report("neural", neural, labels, bank.prevalence)

    (paths.artifacts / "encoder_comparison.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
