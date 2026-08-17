"""Honest transfer estimate for the constant report.

The optimizer fits the report to the references it is scored on, so its own
number is in-sample. The hidden test set is 50 cases from a centre that is not in
the training release at all, so the question that matters is: how much of the
score survives a centre the report has never seen?

Refits from scratch on three centres and scores on the fourth, for each centre in
turn. The gap between in-sample and held-out is the shrinkage to expect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_constant_report import (  # noqa: E402
    LEADER_BLEU,
    LEADER_METEOR,
    ConstantScorer,
    objective,
    render,
)

from cbct_reasoner.data.corpus import load_corpus  # noqa: E402
from cbct_reasoner.prototypes import PrototypeBank  # noqa: E402


def fit(bank, references: list[str], candidates: list[int], max_sentences: int = 30) -> set[int]:
    scorer = ConstantScorer(references)
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
    args = parser.parse_args()

    entries = load_corpus("work/corpus.jsonl")
    bank = PrototypeBank.load("artifacts/prototypes.json")
    candidates = [p.index for p in bank if p.prevalence >= args.min_prevalence]
    centres = sorted({e.center for e in entries})

    print(f"leader targets: BLEU {LEADER_BLEU}  METEOR {LEADER_METEOR}")
    print(f"{'held out':>10} {'n':>4} {'in-sample':>22} {'HELD OUT':>22} {'sentences':>10}")
    rows = []
    for held in centres:
        train = [e.reference for e in entries if e.center != held]
        test = [e.reference for e in entries if e.center == held]
        if len(test) < 20:
            continue
        chosen = fit(bank, train, candidates)
        _, tokens = render(bank, chosen)
        in_b, in_m = ConstantScorer(train).score(tokens)
        out_b, out_m = ConstantScorer(test).score(tokens)
        rows.append((out_b, out_m))
        print(
            f"{held:>10} {len(test):4d} "
            f"{f'{in_b:.4f}/{in_m:.4f}':>22} {f'{out_b:.4f}/{out_m:.4f}':>22} {len(chosen):10d}"
        )

    if rows:
        bleu = np.mean([r[0] for r in rows])
        meteor = np.mean([r[1] for r in rows])
        print(
            f"\nmean held-out: BLEU {bleu:.4f} ({'above' if bleu > LEADER_BLEU else 'below'} "
            f"leader) METEOR {meteor:.4f} ({'above' if meteor > LEADER_METEOR else 'below'} leader)"
        )
        print("This is the number to expect on the hidden test set, not the in-sample fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
