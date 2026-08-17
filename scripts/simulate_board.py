"""Simulate the whole leaderboard after adding a candidate entry.

Mean position is a *relative* score: inserting an entry shifts everyone's rank,
including our own existing one. Judging a candidate by comparing its mean
position against today's 2.0 is wrong, because today's 2.0 is a three-way tie
that a new entry can break.

The right question is what the board looks like afterwards, and whether the new
entry is alone at the top.

Estimator, calibrated on the real result (predicted mean-fold BLEU 0.1429 /
METEOR 0.3490; actual 0.0943 / 0.3542):

* BLEU  -> worst leave-one-centre-out fold (4-gram overlap is centre-specific)
* METEOR -> fold mean (token overlap is stable)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BOARD = [
    ("shivam Stage Seg GPU", 0.1317, 0.3113),
    ("Ryhn tf4 baseline", 0.1102, 0.3191),
    ("DiceMed Prior (ours)", 0.0943, 0.3542),
    ("liolio 2stage", 0.0935, 0.3089),
    ("Zack retrieval", 0.0900, 0.3025),
]


def mean_positions(entries: list[tuple[str, float, float]]) -> dict[str, float]:
    by_bleu = sorted(entries, key=lambda e: -e[1])
    by_meteor = sorted(entries, key=lambda e: -e[2])
    bleu_rank = {e[0]: i + 1 for i, e in enumerate(by_bleu)}
    meteor_rank = {e[0]: i + 1 for i, e in enumerate(by_meteor)}
    return {e[0]: (bleu_rank[e[0]] + meteor_rank[e[0]]) / 2 for e in entries}


def evaluate(bleu: float, meteor: float, label: str = "NEW") -> dict:
    after = mean_positions([*BOARD, (label, bleu, meteor)])
    best = min(after.values())
    winners = [name for name, position in after.items() if position == best]
    return {
        "new_position": after[label],
        "best_position": best,
        "winners": winners,
        "sole_first": winners == [label],
        "shares_first": label in winners,
        "board": after,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, default=Path("artifacts/leaderboard_search.json"))
    args = parser.parse_args()

    before = mean_positions(BOARD)
    tied = [n for n, p in before.items() if p == min(before.values())]
    print("board today:")
    for name, position in sorted(before.items(), key=lambda kv: kv[1]):
        print(f"  {position:4.1f}  {name}")
    print(f"  -> {len(tied)}-way tie for first: {tied}\n")

    rows = json.loads(args.search.read_text(encoding="utf-8"))
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["bleu_weight"], row["aggregate"], row["min_prevalence"]), []
        ).append(row)

    results = []
    for key, folds in grouped.items():
        bleu = min(f["held_out_bleu_4"] for f in folds)
        meteor = sum(f["held_out_meteor"] for f in folds) / len(folds)
        outcome = evaluate(bleu, meteor)
        results.append(
            {
                "bleu_weight": key[0],
                "aggregate": key[1],
                "min_prevalence": key[2],
                "est_bleu_4": bleu,
                "est_meteor": meteor,
                "sentences": sum(f["sentences"] for f in folds) / len(folds),
                **{k: v for k, v in outcome.items() if k != "board"},
            }
        )

    results.sort(key=lambda r: (not r["sole_first"], r["new_position"], -r["est_bleu_4"]))
    header = (
        f"{'w':>4} {'agg':>5} {'prev':>5} {'est BLEU':>9} {'est METEOR':>11} "
        f"{'new pos':>8} {'outcome':>22}"
    )
    print(header)
    print("-" * len(header))
    for row in results[:12]:
        outcome = (
            "SOLE FIRST"
            if row["sole_first"]
            else f"tied first x{len(row['winners'])}"
            if row["shares_first"]
            else f"behind {row['winners'][0][:14]}"
        )
        print(
            f"{row['bleu_weight']:4.1f} {row['aggregate']:>5} {row['min_prevalence']:5.2f} "
            f"{row['est_bleu_4']:9.4f} {row['est_meteor']:11.4f} "
            f"{row['new_position']:8.1f} {outcome:>22}"
        )

    Path("artifacts/board_simulation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    sole = [r for r in results if r["sole_first"]]
    print()
    if sole:
        best = sole[0]
        print(
            f"SUBMIT  w={best['bleu_weight']} {best['aggregate']} "
            f"prev={best['min_prevalence']}  ->  sole first at mean position "
            f"{best['new_position']}"
        )
        detail = evaluate(best["est_bleu_4"], best["est_meteor"])["board"]
        for name, position in sorted(detail.items(), key=lambda kv: kv[1]):
            print(f"    {position:4.1f}  {name}")
    else:
        print("No configuration takes sole first; the best available only ties.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
