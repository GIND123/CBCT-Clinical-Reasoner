"""Choose the final submission using an estimator calibrated on a real result.

The shipped constant report predicted held-out BLEU 0.1429 (mean over four
leave-one-centre-out folds) and scored 0.0943. Its worst folds were 0.094 and
0.099. METEOR predicted 0.3490 and scored 0.3542, essentially the fold mean.

So the two metrics transfer differently, and the estimator has to reflect that:

* **BLEU is estimated by the worst fold.** Exact 4-gram overlap is
  centre-specific, and the hidden centre behaves like the hardest one seen.
* **METEOR is estimated by the fold mean.** Token-level overlap is stable across
  centres.

Ranking also has to account for the fact that a new entry competes with our own
existing one: pushing a rival down on BLEU also pushes DiceMed Prior down, so a
submission is only worth making if it beats mean position 2.0 by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Current board, excluding our own entry, as (BLEU-4, METEOR).
RIVALS = {
    "shivam Stage Seg GPU": (0.1317, 0.3113),
    "Ryhn tf4 baseline": (0.1102, 0.3191),
    "liolio 2stage": (0.0935, 0.3089),
    "Zack retrieval": (0.0900, 0.3025),
}
#: Our existing submission, which stays on the board and competes with any new one.
OURS = ("DiceMed Prior", 0.0943, 0.3542)
CURRENT_MEAN_POSITION = 2.0


def positions(bleu: float, meteor: float) -> tuple[int, int, float]:
    field = list(RIVALS.values()) + [(OURS[1], OURS[2])]
    bleu_rank = 1 + sum(1 for b, _ in field if b > bleu)
    meteor_rank = 1 + sum(1 for _, m in field if m > meteor)
    return bleu_rank, meteor_rank, (bleu_rank + meteor_rank) / 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, default=Path("artifacts/leaderboard_search.json"))
    args = parser.parse_args()

    rows = json.loads(args.search.read_text(encoding="utf-8"))
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["bleu_weight"], row["aggregate"], row["min_prevalence"]), []
        ).append(row)

    summary = []
    for key, folds in grouped.items():
        bleu_estimate = min(f["held_out_bleu_4"] for f in folds)
        meteor_estimate = sum(f["held_out_meteor"] for f in folds) / len(folds)
        bleu_rank, meteor_rank, mean_position = positions(bleu_estimate, meteor_estimate)
        summary.append(
            {
                "bleu_weight": key[0],
                "aggregate": key[1],
                "min_prevalence": key[2],
                "est_bleu_4": bleu_estimate,
                "est_meteor": meteor_estimate,
                "fold_mean_bleu_4": sum(f["held_out_bleu_4"] for f in folds) / len(folds),
                "bleu_rank": bleu_rank,
                "meteor_rank": meteor_rank,
                "mean_position": mean_position,
                "improves_on_current": mean_position < CURRENT_MEAN_POSITION,
                "sentences": sum(f["sentences"] for f in folds) / len(folds),
                "tokens": sum(f["tokens"] for f in folds) / len(folds),
            }
        )

    summary.sort(key=lambda r: (r["mean_position"], -r["est_bleu_4"]))
    header = (
        f"{'w':>4} {'agg':>5} {'prev':>5} {'est BLEU':>9} {'est METEOR':>11} "
        f"{'ranks':>7} {'mean':>5} {'sent':>5} {'better?':>8}"
    )
    print(f"estimator: BLEU = worst fold, METEOR = fold mean (calibrated on the real result)")
    print(
        f"our current entry: BLEU {OURS[1]} METEOR {OURS[2]} -> mean position {CURRENT_MEAN_POSITION}\n"
    )
    print(header)
    print("-" * len(header))
    for row in summary[:16]:
        print(
            f"{row['bleu_weight']:4.1f} {row['aggregate']:>5} {row['min_prevalence']:5.2f} "
            f"{row['est_bleu_4']:9.4f} {row['est_meteor']:11.4f} "
            f"{str(row['bleu_rank']) + '/' + str(row['meteor_rank']):>7} "
            f"{row['mean_position']:5.1f} {row['sentences']:5.0f} "
            f"{'YES' if row['improves_on_current'] else 'no':>8}"
        )

    Path("artifacts/submission_selection.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    winners = [r for r in summary if r["improves_on_current"]]
    print()
    if winners:
        best = winners[0]
        print(
            f"SUBMIT: w={best['bleu_weight']} {best['aggregate']} prev={best['min_prevalence']} "
            f"-> expected mean position {best['mean_position']} "
            f"(BLEU rank {best['bleu_rank']}, METEOR rank {best['meteor_rank']})"
        )
    else:
        print(
            "NO CONFIGURATION BEATS THE CURRENT ENTRY.\n"
            "Submitting anyway would add a row that pushes rivals down on BLEU and "
            "therefore pushes DiceMed Prior down too, costing the tie for first."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
