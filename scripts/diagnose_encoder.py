"""Did the encoder learn to read the image, or is it reproducing the prior?

Mean average precision over a thousand-statement label space is a poor answer:
most prototypes appear in a handful of cases, so the average is dominated by
columns where nothing is learnable and it barely moves whatever the model does.

This asks the question directly, per prototype:

* **AUC against the prior.** The prior assigns every case the same probability,
  so its AUC is exactly 0.5 by construction. Anything above that is information
  read from the image.
* **Grouped by prevalence.** A statement seen in three cases cannot be learned
  from 500 training volumes; one seen in 300 can. Separating them shows whether
  the encoder is learning where learning is possible.
* **Weighted by prevalence**, because those are the statements the decoder
  actually emits and therefore the ones that move the score.

Run after `train`:  python scripts/diagnose_encoder.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_reasoner.config import default_paths  # noqa: E402
from cbct_reasoner.prototypes import PrototypeBank, load_labels  # noqa: E402


def auc(scores: np.ndarray, targets: np.ndarray) -> float | None:
    """Rank-based AUC; ``None`` when a column is all-positive or all-negative."""
    positives = targets.sum()
    if positives == 0 or positives == len(targets):
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties so a constant predictor scores exactly 0.5.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    negatives = len(targets) - positives
    return float(
        (ranks[targets == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()

    paths = default_paths()
    if args.work:
        paths = paths.with_root(args.work)

    bank = PrototypeBank.load(paths.prototypes)
    case_ids, labels = load_labels(paths.labels)
    with np.load(paths.oof, allow_pickle=False) as archive:
        oof_ids = [str(v) for v in archive["case_ids"].tolist()]
        probabilities = archive["probabilities"].astype(np.float64)

    index = {case_id: row for row, case_id in enumerate(case_ids)}
    aligned = np.stack([labels[index[case_id]] for case_id in oof_ids])

    prevalence = bank.prevalence.astype(np.float64)
    rows = []
    for column in range(len(bank)):
        value = auc(probabilities[:, column], aligned[:, column])
        if value is not None:
            rows.append((column, prevalence[column], int(aligned[:, column].sum()), value))

    if not rows:
        print("No evaluable prototypes.")
        return 1

    aucs = np.asarray([r[3] for r in rows])
    supports = np.asarray([r[2] for r in rows])
    weights = np.asarray([r[1] for r in rows])

    print(f"evaluable prototypes: {len(rows)} / {len(bank)}\n")
    print("A prior-only model scores exactly 0.500 by construction.\n")
    print(f"{'support band':>16} {'n':>5} {'mean AUC':>9} {'>0.55':>7} {'>0.60':>7}")
    for low, high, name in [
        (1, 5, "1-5 cases"),
        (6, 20, "6-20 cases"),
        (21, 60, "21-60 cases"),
        (61, 10**6, "60+ cases"),
    ]:
        mask = (supports >= low) & (supports <= high)
        if not mask.any():
            continue
        band = aucs[mask]
        print(
            f"{name:>16} {mask.sum():5d} {band.mean():9.3f} "
            f"{(band > 0.55).mean():7.1%} {(band > 0.60).mean():7.1%}"
        )

    weighted = float((aucs * weights).sum() / weights.sum())
    print(f"\nunweighted mean AUC : {aucs.mean():.4f}")
    print(f"prevalence-weighted : {weighted:.4f}   <- the statements the decoder emits")

    top = sorted(rows, key=lambda r: -r[3])[:10]
    print("\nbest-learned statements:")
    for column, prev, support, value in top:
        print(f"  AUC {value:.3f}  n={support:3d}  prev={prev:.2f}  {bank[column].text[:78]}")

    payload = {
        "evaluable": len(rows),
        "mean_auc": float(aucs.mean()),
        "prevalence_weighted_auc": weighted,
        "fraction_above_0_55": float((aucs > 0.55).mean()),
        "fraction_above_0_60": float((aucs > 0.60).mean()),
    }
    (paths.artifacts / "encoder_diagnosis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
