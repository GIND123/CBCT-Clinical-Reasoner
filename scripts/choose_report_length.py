"""Pick how many statements the report should make.

The Final Score keeps rising as statements are added, so left alone the optimizer
returns whatever its cap allows. That is the wrong stopping rule, because the
Final Score is a filter and not the prize:

* **The Final Score decides who reaches the review.** More statements raise it -
  recall climbs, and precision holds up because the statements being added are
  ones that are true of most scans.
* **A double-blind clinical preference review decides the ranking.** There, a
  report is read by a surgeon beside a competitor's report for the same case.

Those pull in opposite directions, and the second one is unforgiving. Reference
reports in this corpus carry a mean of 9.8 phrases; at 26 statements the report
is nearly three times a real one, and about eleven of its statements are false
for any given patient - more false statements than a real report makes
statements at all. No reader misses that.

So the length is chosen from the curve rather than from a cap: take the knee,
where the Final Score has stopped buying much per statement. Everything after it
costs credibility at a steep discount.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Reference reports average this many phrases (median 9, p90 15).
REFERENCE_PHRASES = 9.8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default="artifacts/final_score_report_v2_trajectory.json")
    parser.add_argument(
        "--statements",
        type=int,
        default=15,
        help="pin the report length; 0 falls back to the --min-gain knee",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.004,
        help="when --statements is 0, stop once a statement buys less than this",
    )
    parser.add_argument("--out", default="artifacts/report_length_choice.json")
    parser.add_argument("--core-out", default="artifacts/final_score_core.json")
    args = parser.parse_args()

    trajectory = json.loads(Path(args.trajectory).read_text(encoding="utf-8"))
    if not trajectory:
        print("empty trajectory")
        return 1

    header = (
        f"{'n':>3} {'FINAL':>7} {'gain':>7} {'P':>6} {'R':>6} "
        f"{'false/case':>10} {'BLEU':>7} {'METEOR':>7} {'vs ref':>7}"
    )
    print(header)
    print("-" * len(header))
    knee = None
    for position, row in enumerate(trajectory):
        gain = row["final"] - (trajectory[position - 1]["final"] if position else 0.0)
        if knee is None and position and gain < args.min_gain:
            knee = trajectory[position - 1]
        print(
            f"{row['statements']:3d} {row['final']:7.4f} {gain:+7.4f} "
            f"{row['logical_precision']:6.3f} {row['logical_recall']:6.3f} "
            f"{row['false_statements_per_case']:10.1f} {row['bleu_4']:7.4f} "
            f"{row['meteor']:7.4f} {row['statements'] / REFERENCE_PHRASES:6.1f}x"
        )

    best = trajectory[-1]
    if knee is None:
        knee = best
    # The marginal gain decays smoothly rather than falling off a cliff, so a
    # fixed threshold picks a length by accident. When a length is specified it
    # wins, and the threshold's answer is still printed above for comparison.
    if args.statements:
        pinned = next((r for r in trajectory if r["statements"] == args.statements), None)
        if pinned is None:
            print(f"\nno prefix of {args.statements} statements in the trajectory")
            return 1
        print(f"\nthreshold knee would be {knee['statements']} statements")
        knee = pinned
    print(
        f"\nknee at {knee['statements']} statements: FINAL {knee['final']:.4f}, "
        f"{knee['false_statements_per_case']:.1f} false statements per case, "
        f"{knee['statements'] / REFERENCE_PHRASES:.1f}x a reference report"
    )
    print(
        f"running to {best['statements']} would add {best['final'] - knee['final']:+.4f} FINAL "
        f"and {best['false_statements_per_case'] - knee['false_statements_per_case']:+.1f} "
        f"false statements per case"
    )

    Path(args.out).write_text(
        json.dumps({"knee": knee, "longest": best, "min_gain": args.min_gain}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Written in the shape fit_adaptive_report.py consumes, so the chosen length
    # flows straight into gating without a hand-copied index list.
    from cbct_reasoner.prototypes import PrototypeBank
    from cbct_reasoner.text import join_report

    bank = PrototypeBank.load("artifacts/prototypes.json")
    text = join_report([bank[i].text for i in knee["indices"]])
    core_path = Path(args.core_out)
    core_path.write_text(
        json.dumps({**knee, "report": text}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(core_path.with_suffix(".txt")).write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {args.out} and {core_path}")
    print(f"\n{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
