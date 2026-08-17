"""Measure true RadFact for the candidate reports and check the surrogate against it.

RadFact is 80% of the Final Score that decides which submissions reach the
double-blind clinical review, and every clinical number in this repository so far
came from an offline lexical surrogate. This runs the organizers' own
``radfact_lite`` on Modal against a locally served LLM, then reports how well the
surrogate tracked it — because a surrogate that ranks candidates differently from
the real metric has been optimizing the wrong thing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cbct_reasoner.data.corpus import load_corpus  # noqa: E402
from cbct_reasoner.decode.decoder import ReportDecoder  # noqa: E402
from cbct_reasoner.metrics.score import score_reports  # noqa: E402
from cbct_reasoner.prototypes import PrototypeBank  # noqa: E402


def candidates() -> dict[str, str]:
    bank = PrototypeBank.load("artifacts/prototypes.json")
    out: dict[str, str] = {}
    prior_decoder = ReportDecoder.load("artifacts/runs/neural/decoder.json", bank)
    out["prior"] = prior_decoder.decode(bank.prevalence)
    shallow_decoder = ReportDecoder.load("artifacts/runs/shallow/decoder.json", bank)
    out["shallow_prototypes"] = shallow_decoder.decode(bank.prevalence)
    captioning = Path("artifacts/final_report.txt")
    if captioning.is_file():
        out["captioning_optimised"] = captioning.read_text(encoding="utf-8").strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    reports = candidates()
    entries = load_corpus("work/corpus.jsonl")
    references = {e.case_id: e.reference for e in entries}

    print("surrogate scores (all 622 cases):")
    surrogate = {}
    for name, text in reports.items():
        score = score_reports({c: text for c in references}, references)
        surrogate[name] = score.to_dict()
        print(
            f"  {name:<24} final {score.final:.4f}  clinical {score.clinical:.4f} "
            f"(P {score.logical_precision:.3f} R {score.logical_recall:.3f})"
        )

    import modal

    fn = modal.Function.from_name("cbct-clinical-reasoner", "radfact_eval")
    print(f"\nrunning real RadFact on {args.limit} cases with {args.model} ...")
    result = fn.remote({"reports": reports, "limit": args.limit, "model": args.model})

    print("\nreal RadFact:")
    rows = []
    for name, aggregate in result["results"].items():
        caption = surrogate[name]
        real_final = 0.8 * aggregate["logical_f1"] + 0.2 * caption["captioning"]
        rows.append((name, aggregate, real_final))
        print(
            f"  {name:<24} F1 {aggregate['logical_f1']:.4f} "
            f"(P {aggregate['logical_precision']:.3f} R {aggregate['logical_recall']:.3f})"
            f"  ->  FINAL {real_final:.4f}   [surrogate said {caption['final']:.4f}]"
        )

    order_real = [r[0] for r in sorted(rows, key=lambda r: -r[2])]
    order_surrogate = sorted(surrogate, key=lambda n: -surrogate[n]["final"])
    print(f"\nranking by real metric : {order_real}")
    print(f"ranking by surrogate   : {order_surrogate}")
    print(f"surrogate picks the same winner: {order_real[0] == order_surrogate[0]}")

    Path("artifacts/surrogate_validation.json").write_text(
        json.dumps(
            {
                "model": result["model"],
                "cases": result["cases"],
                "surrogate": surrogate,
                "real": result["results"],
                "real_final": {name: value for name, _, value in rows},
                "same_winner": order_real[0] == order_surrogate[0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
