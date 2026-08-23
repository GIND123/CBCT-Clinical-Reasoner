"""Score every predictor under every fitted threshold vector.

Coordinate ascent is greedy, so a predictor's *own* calibration can land in a
local optimum that another run's thresholds beat - which is exactly what happened
here: the corpus prior scored 0.324 under its own ascent and 0.358 under the
thresholds fitted for the neural model.

Comparing predictors by their own-calibration score alone therefore measures the
search as much as the model. This scores the full cross-product on one scorer,
one label space, one decoder, and reports each predictor's best achievable
result - the number that should actually decide what ships.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_reasoner.config import ExperimentConfig, default_paths  # noqa: E402
from cbct_reasoner.data.corpus import load_corpus  # noqa: E402
from cbct_reasoner.decode.calibrate import CalibrationScorer  # noqa: E402
from cbct_reasoner.decode.decoder import DecoderSettings, ReportDecoder  # noqa: E402
from cbct_reasoner.prototypes import PrototypeBank  # noqa: E402


def load_probabilities(path: Path, case_ids: list[str], fallback: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        ids = [str(v) for v in archive["case_ids"].tolist()]
        matrix = archive["probabilities"].astype(np.float32)
    index = {case_id: row for row, case_id in enumerate(ids)}
    return np.stack([matrix[index[c]] if c in index else fallback for c in case_ids]).astype(
        np.float32
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/toothfairy4.json"))
    args = parser.parse_args()

    paths = default_paths()
    config = ExperimentConfig.load(args.config if args.config.is_file() else None)
    bank = PrototypeBank.load(paths.prototypes)
    entries = load_corpus(paths.corpus)
    case_ids = [entry.case_id for entry in entries]

    scorer = CalibrationScorer(
        bank,
        [entry.reference for entry in entries],
        reference_phrases=[entry.phrases for entry in entries],
        clinical_weight=config.decode.clinical_weight,
        captioning_weight=config.decode.captioning_weight,
    )
    settings = DecoderSettings(
        min_sentences=config.decode.min_sentences, max_sentences=config.decode.max_sentences
    )

    prior = np.tile(bank.prevalence.astype(np.float32), (len(case_ids), 1))
    predictors: dict[str, np.ndarray] = {"prior": prior}
    for name, path in (
        ("neural", paths.work / "oof.npz"),
        ("shallow", paths.work / "oof_shallow.npz"),
    ):
        if path.is_file():
            predictors[name] = load_probabilities(path, case_ids, bank.prevalence)

    thresholds: dict[str, np.ndarray] = {}
    for name, path in (
        ("neural", paths.artifacts / "runs" / "neural" / "decoder.json"),
        ("shallow", paths.artifacts / "decoder.json"),
        ("prior", paths.artifacts / "runs" / "prior" / "decoder.json"),
    ):
        if path.is_file():
            thresholds[name] = np.asarray(
                json.loads(path.read_text(encoding="utf-8"))["thresholds"], dtype=np.float32
            )

    print(f"predictors: {list(predictors)}   threshold sets: {list(thresholds)}\n")
    header = "predictor    " + "".join(f"{'via ' + n:>16}" for n in thresholds) + f"{'BEST':>10}"
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, float]] = {}
    best_overall = ("", "", -1.0)
    for predictor, probabilities in predictors.items():
        row: dict[str, float] = {}
        for name, vector in thresholds.items():
            decoder = ReportDecoder(bank, vector, settings=settings)
            row[name] = scorer.score_selection(decoder.select_many(probabilities)).final
        best_name = max(row, key=row.__getitem__)
        results[predictor] = row
        if row[best_name] > best_overall[2]:
            best_overall = (predictor, best_name, row[best_name])
        cells = "".join(f"{row[n]:16.4f}" for n in thresholds)
        print(f"{predictor:<13}{cells}{row[best_name]:10.4f}")

    predictor, threshold_name, value = best_overall
    print(f"\nBEST: {predictor} probabilities with {threshold_name} thresholds -> {value:.4f}")

    decoder = ReportDecoder(bank, thresholds[threshold_name], settings=settings)
    breakdown = scorer.score_selection(decoder.select_many(predictors[predictor]))
    for key, item in breakdown.to_dict().items():
        print(f"  {key:20s} {item:.4f}")

    (paths.artifacts / "predictor_comparison.json").write_text(
        json.dumps(
            {
                "grid": results,
                "best": {
                    "predictor": predictor,
                    "thresholds": threshold_name,
                    **breakdown.to_dict(),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
