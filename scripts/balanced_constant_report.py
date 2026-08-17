"""Fit the constant report with centres weighted equally, and validate transfer.

Pooling every reference lets centre P dominate: it is 412 of 622 cases, so the
fitted report drifts toward P's dictation style. That is the wrong target when
the hidden test set is 50 cases from a centre in the training data at all — and
it shows, held-out transfer to centre A being by far the worst fold.

Weighting each centre equally optimizes for "works everywhere" instead of "works
for the biggest centre", which is the closest available proxy for an unseen one.

Compares pooled against centre-balanced fitting under several BLEU/METEOR
weightings, scoring each on a fully held-out centre.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_constant_report import (  # noqa: E402
    LEADER_BLEU,
    LEADER_METEOR,
    ConstantScorer,
    render,
)

from cbct_reasoner.data.corpus import load_corpus  # noqa: E402
from cbct_reasoner.prototypes import PrototypeBank  # noqa: E402


class GroupScorer:
    """Mean of per-centre (BLEU, METEOR), so every centre counts the same."""

    def __init__(self, groups: list[list[str]]) -> None:
        self.scorers = [ConstantScorer(g) for g in groups if g]

    def score(self, tokens: list[str]) -> tuple[float, float]:
        pairs = [s.score(tokens) for s in self.scorers]
        return (
            float(np.mean([p[0] for p in pairs])),
            float(np.mean([p[1] for p in pairs])),
        )


def fit(bank, scorer, candidates, objective, max_sentences=32) -> set[int]:
    chosen: set[int] = set()
    best = 0.0
    for _ in range(max_sentences):
        options = []
        for index in candidates:
            if index in chosen:
                continue
            _, tokens = render(bank, chosen | {index})
            options.append((objective(*scorer.score(tokens)), index))
        if not options:
            break
        value, index = max(options)
        if value <= best + 1e-6:
            break
        chosen.add(index)
        best = value
    for _ in range(2):
        improved = False
        for index in candidates:
            trial = (chosen - {index}) if index in chosen else (chosen | {index})
            if not trial:
                continue
            _, tokens = render(bank, trial)
            value = objective(*scorer.score(tokens))
            if value > best + 1e-6:
                chosen, best, improved = trial, value, True
        if not improved:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-prevalence", type=float, default=0.015)
    parser.add_argument("--weights", type=float, nargs="*", default=[0.4, 0.25])
    args = parser.parse_args()

    entries = load_corpus("work/corpus.jsonl")
    bank = PrototypeBank.load("artifacts/prototypes.json")
    candidates = [p.index for p in bank if p.prevalence >= args.min_prevalence]
    centres = sorted({e.center for e in entries})

    print(f"leader: BLEU {LEADER_BLEU} METEOR {LEADER_METEOR}")
    print(f"{'mode':>10} {'w_bleu':>7} {'held-out BLEU':>15} {'held-out METEOR':>17} {'both':>6}")

    results: dict[str, dict] = {}
    for weight in args.weights:

        def objective(bleu: float, meteor: float, w: float = weight) -> float:
            return w * bleu / LEADER_BLEU + (1 - w) * meteor / LEADER_METEOR

        for mode in ("pooled", "balanced"):
            pairs = []
            for held in centres:
                groups = [
                    [e.reference for e in entries if e.center == c] for c in centres if c != held
                ]
                scorer = (
                    ConstantScorer([r for g in groups for r in g])
                    if mode == "pooled"
                    else GroupScorer(groups)
                )
                chosen = fit(bank, scorer, candidates, objective)
                _, tokens = render(bank, chosen)
                test = [e.reference for e in entries if e.center == held]
                pairs.append(ConstantScorer(test).score(tokens))
            bleu = float(np.mean([p[0] for p in pairs]))
            meteor = float(np.mean([p[1] for p in pairs]))
            wins = bleu > LEADER_BLEU and meteor > LEADER_METEOR
            results[f"{mode}_w{weight}"] = {
                "bleu": bleu,
                "meteor": meteor,
                "beats_both": wins,
                "per_fold": pairs,
            }
            print(f"{mode:>10} {weight:7.2f} {bleu:15.4f} {meteor:17.4f} {str(wins):>6}")

    Path("artifacts/balanced_sweep.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
