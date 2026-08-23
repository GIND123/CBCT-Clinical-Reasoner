"""How well does the offline surrogate measure clinical agreement?

Every selection decision in this repository is made against a lexical stand-in
for RadFact, so the stand-in's quality bounds everything. Validating it against
the real metric needs a GPU and an LLM; this needs neither, and answers a
sharper question: **can the surrogate tell a report about this patient from a
report about a different one?**

The candidate is each case's **oracle prototype report** - the statements the
decoder would emit if it predicted that case's labels perfectly. Scoring the
reference against itself would be a tautology; the oracle report is the same
clinical content in the corpus's own stock phrasings, which is exactly the
regime the decoder operates in.

Each oracle report is scored against its own reference (matched) and against
other cases' references (mismatched). A metric that measures clinical agreement
scores matched pairs higher. A metric that mostly measures how radiology reports
are worded scores them alike, because these reports are all assembled from the
same small stock of sentences.

Reported as AUC of the matched/mismatched separation. Chance is 0.5, and the
number is comparable across ontology changes, which is what makes it useful as a
regression check on the surrogate itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cbct_reasoner.data.corpus import load_corpus  # noqa: E402
from cbct_reasoner.metrics.radfact import LexicalRadFact  # noqa: E402
from cbct_reasoner.prototypes import PrototypeBank, load_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=250)
    parser.add_argument("--decoys", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--label", default="current")
    parser.add_argument("--out", default="artifacts/surrogate_quality.json")
    args = parser.parse_args()

    entries = load_corpus("work/corpus.jsonl")
    bank = PrototypeBank.load("artifacts/prototypes.json")
    case_ids, labels = load_labels("work/labels.npz")
    reference_of = {entry.case_id: entry.phrases for entry in entries}
    order = {index: position for position, index in enumerate(bank.render_order)}

    usable = [i for i, case in enumerate(case_ids) if case in reference_of and labels[i].any()]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(usable, size=min(args.cases, len(usable)), replace=False)

    def oracle_report(row: int) -> list[str]:
        indices = np.flatnonzero(labels[row])
        return [bank[i].text for i in sorted(indices, key=lambda i: order.get(i, len(order)))]

    metric = LexicalRadFact()
    matched, mismatched = [], []
    for position, index in enumerate(chosen):
        candidate = oracle_report(index)
        matched.append(metric.score_case("m", candidate, reference_of[case_ids[index]]).logical_f1)
        for other in rng.choice(usable, size=args.decoys, replace=False):
            if other == index:
                continue
            reference = reference_of[case_ids[other]]
            mismatched.append(metric.score_case("x", candidate, reference).logical_f1)
        if (position + 1) % 50 == 0:
            print(f"  {position + 1}/{len(chosen)} cases", flush=True)

    matched_array = np.asarray(matched)
    mismatched_array = np.asarray(mismatched)
    # AUC as the probability a matched pair outscores a mismatched one, ties half.
    comparison = matched_array[:, None] - mismatched_array[None, :]
    auc = float((comparison > 0).mean() + 0.5 * (comparison == 0).mean())

    print(f"\nsurrogate discrimination ({args.label})")
    print(f"  matched    mean F1 {matched_array.mean():.4f}  (n={matched_array.size})")
    print(f"  mismatched mean F1 {mismatched_array.mean():.4f}  (n={mismatched_array.size})")
    print(f"  separation         {matched_array.mean() - mismatched_array.mean():+.4f}")
    print(f"  AUC                {auc:.4f}   (0.5 = the metric cannot tell them apart)")

    destination = Path(args.out)
    history = {}
    if destination.is_file():
        history = json.loads(destination.read_text(encoding="utf-8"))
    history[args.label] = {
        "auc": auc,
        "matched_mean": float(matched_array.mean()),
        "mismatched_mean": float(mismatched_array.mean()),
        "cases": int(matched_array.size),
    }
    destination.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
