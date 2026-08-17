"""Measure every predictor end to end, through the same scoring path.

Each predictor is decoded with the thresholds that scored best for it
out-of-fold, then scored with ``score_reports`` - the same function
``cbct-reasoner evaluate`` uses - so the numbers here are directly comparable to
the reported result rather than reconstructed from intermediate breakdowns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_reasoner.config import default_paths  # noqa: E402
from cbct_reasoner.data.corpus import load_corpus  # noqa: E402
from cbct_reasoner.decode.decoder import ReportDecoder  # noqa: E402
from cbct_reasoner.metrics.score import score_reports  # noqa: E402
from cbct_reasoner.prototypes import PrototypeBank  # noqa: E402

#: (label, out-of-fold probabilities, threshold vector fitted for them)
VARIANTS = (
    ("corpus prior (no imaging)", None, "artifacts/runs/neural/decoder.json"),
    ("linear, 122 features", "work/oof_shallow.npz", "artifacts/runs/shallow/decoder.json"),
    ("fine-tuned encoder (29M params)", "work/oof.npz", "artifacts/runs/neural/decoder.json"),
)


def main() -> int:
    paths = default_paths()
    bank = PrototypeBank.load(paths.prototypes)
    entries = load_corpus(paths.corpus)
    case_ids = [entry.case_id for entry in entries]
    references = {entry.case_id: entry.reference for entry in entries}

    results: dict[str, dict[str, float]] = {}
    for label, oof_path, decoder_path in VARIANTS:
        if oof_path is None:
            probabilities = np.tile(bank.prevalence.astype(np.float32), (len(case_ids), 1))
            ids = case_ids
        else:
            path = Path(oof_path)
            if not path.is_file():
                print(f"skipping {label}: {path} missing")
                continue
            with np.load(path, allow_pickle=False) as archive:
                ids = [str(v) for v in archive["case_ids"].tolist()]
                probabilities = archive["probabilities"].astype(np.float32)

        decoder = ReportDecoder.load(Path(decoder_path), bank)
        predictions = {
            case_id: report
            for case_id, report in zip(ids, decoder.decode_batch(probabilities), strict=True)
        }
        score = score_reports(predictions, {k: references[k] for k in predictions})
        results[label] = {
            key: value for key, value in score.to_dict().items() if isinstance(value, (int, float))
        }
        print(
            f"{label:<34} final={score.final:.4f} clinical={score.clinical:.4f} "
            f"bleu={score.bleu_4:.4f} meteor={score.meteor:.4f}"
        )

    Path("artifacts/ablation.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    from cbct_reasoner import plots

    written = plots.plot_ablation(list(results.items()), Path("artifacts/plots"))
    print("figure:", [str(p) for p in written])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
