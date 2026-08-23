"""Which per-case signals actually predict which statements?

Gating only pays when the gate is right more often than the base rate, so before
spending inference-time risk on a signal it is worth measuring what each one
buys. Three feature sets are compared by cross-validated AUC per statement:

``geometry``
    Nine numbers from the image header - volume dimensions, voxel spacing, and
    physical extent. Free, and cannot fail at inference time.
``intensity``
    Two numbers already recorded during preprocessing: the 99.5th intensity
    percentile and the foreground fraction. The percentile is a proxy for
    high-density material, so it is the natural candidate for statements about
    restorations, crowns, and endodontic treatment.
``both``
    The union, which is what a gate would use if intensity earns its place.

The comparison matters because intensity is not free: reading it at inference
time means decoding pixel data, which is exactly the class of failure the
header-only path was built to avoid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cbct_reasoner.data.splits import SplitPlan  # noqa: E402
from cbct_reasoner.prototypes import PrototypeBank, load_labels  # noqa: E402


def load_features(case_ids) -> dict[str, np.ndarray]:
    geometry, intensity = [], []
    for case in case_ids:
        meta = json.loads(Path(f"work/cache/{case}.json").read_text(encoding="utf-8"))
        geometry.append(
            np.concatenate(
                [
                    np.log1p(np.asarray(meta["original_shape_zyx"], float)),
                    np.asarray(meta["original_spacing_zyx"], float),
                    np.log1p(np.asarray(meta["physical_size_mm"], float)),
                ]
            )
        )
        intensity.append(
            [
                np.log1p(max(0.0, float(meta["intensity_high"]))),
                float(meta["intensity_low"]),
                float(meta["foreground_fraction"]),
            ]
        )
    geometry = np.asarray(geometry, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    return {
        "geometry": geometry,
        "intensity": intensity,
        "both": np.hstack([geometry, intensity]),
    }


def cross_validated_auc(features: np.ndarray, target: np.ndarray, folds) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    predictions = np.zeros(len(target), dtype=np.float64)
    for validation in folds:
        train = [i for i in range(len(target)) if i not in set(validation)]
        if target[train].sum() < 3 or target[train].sum() == len(train):
            predictions[validation] = target[train].mean()
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, class_weight="balanced"),
        )
        model.fit(features[train], target[train])
        predictions[validation] = model.predict_proba(features[validation])[:, 1]
    if target.sum() in (0, len(target)):
        return float("nan")
    return float(roc_auc_score(target, predictions))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--min-prevalence", type=float, default=0.03)
    args = parser.parse_args()

    bank = PrototypeBank.load("artifacts/prototypes.json")
    case_ids, labels = load_labels("work/labels.npz")
    plan = SplitPlan.load("work/folds.json")
    position_of = {c: i for i, c in enumerate(case_ids)}
    folds = [[position_of[c] for c in fold.validation if c in position_of] for fold in plan]
    feature_sets = load_features(case_ids)

    candidates = [
        p for p in sorted(bank, key=lambda p: -p.prevalence) if p.prevalence >= args.min_prevalence
    ][: args.top]
    print(f"{len(case_ids)} cases | {len(candidates)} statements | {len(folds)} folds\n")

    rows = []
    for prototype in candidates:
        target = labels[:, prototype.index].astype(int)
        scores = {
            name: cross_validated_auc(features, target, folds)
            for name, features in feature_sets.items()
        }
        rows.append((prototype, scores))

    header = f"{'prev':>5} {'geom':>6} {'inten':>6} {'both':>6}  statement"
    print(header)
    print("-" * 100)
    for prototype, scores in sorted(rows, key=lambda r: -max(r[1].values())):
        text = " ".join(prototype.text.split())
        print(
            f"{prototype.prevalence:5.2f} {scores['geometry']:6.3f} "
            f"{scores['intensity']:6.3f} {scores['both']:6.3f}  {text[:70]}"
        )

    print()
    for name in feature_sets:
        values = [s[name] for _, s in rows if not np.isnan(s[name])]
        strong = sum(1 for v in values if v >= 0.70)
        print(
            f"{name:>9}: mean AUC {np.mean(values):.3f} | {strong} of {len(values)} at AUC >= 0.70"
        )

    gain = [s["both"] - s["geometry"] for _, s in rows if not np.isnan(s["both"])]
    print(f"\nadding intensity moves mean AUC by {np.mean(gain):+.4f}")
    print(f"  statements improved by >= 0.02 AUC: {sum(1 for g in gain if g >= 0.02)}")

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/gate_signal_probe.json").write_text(
        json.dumps(
            [
                {
                    "index": p.index,
                    "prevalence": p.prevalence,
                    "text": " ".join(p.text.split()),
                    **{k: (None if np.isnan(v) else v) for k, v in s.items()},
                }
                for p, s in rows
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
