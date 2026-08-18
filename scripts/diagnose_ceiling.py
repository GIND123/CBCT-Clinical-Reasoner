"""Why no model trained on this release will win the board's BLEU column.

Three approaches were tried and measured, and the reason all three fail is the
same one. It is worth writing down, because it is a property of the data rather
than of any particular model.

**Sentence choice is house style, not image content.** Predicting which
prototype sentence a report uses from *centre identity alone* reaches mean AUC
0.718 over the 60 most frequent sentences, and 72.2% of a sentence's usage sits
inside its single most-used centre. The image-derived model reaches 0.663 on the
same sentences - worse than knowing nothing except which hospital wrote it.

That reframes the earlier result that the encoder was "at chance". The encoder
was not failing to see the anatomy; it was being asked to predict a reporting
convention, which is not in the pixels.

The consequences line up exactly with what was observed:

* A perfect per-case selector scores BLEU 0.4086 against the constant report's
  0.1663, but that gap is mostly the reward for reproducing one centre's
  phrasing verbatim.
* The hidden test centre has its own conventions, absent from all four training
  centres, so the gap is unreachable there. Test BLEU came in at 0.0943, below
  every leave-one-centre-out fold (0.1241 to 0.1968).
* Clustering by acquisition geometry to condition the report loses on both
  metrics at every K, because it splits an already small corpus without
  capturing the variable that matters.
* Perfect knowledge of which teeth are absent - the most visually obvious
  content in the scan - is worth +0.0041 BLEU. The n-grams are in the sentence
  frames, not the tooth numbers.

Run this to reproduce the diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cbct_reasoner.data.corpus import load_corpus  # noqa: E402
from cbct_reasoner.prototypes import PrototypeBank, load_labels  # noqa: E402


def main() -> int:
    from sklearn.metrics import roc_auc_score

    entries = load_corpus("work/corpus.jsonl")
    bank = PrototypeBank.load("artifacts/prototypes.json")
    case_ids, labels = load_labels("work/labels.npz")
    position = {c: i for i, c in enumerate(case_ids)}

    centre = np.array([e.center for e in entries])
    rows = np.array([position[e.case_id] for e in entries])
    targets = labels[rows]
    centres = sorted(set(centre))
    print(f"centres {centres} sizes {[int((centre == c).sum()) for c in centres]}")

    top = [p.index for p in sorted(bank, key=lambda p: -p.prevalence)[:60]]
    onehot = np.stack([(centre == c).astype(float) for c in centres], axis=1)
    style, share = [], []
    for column in top:
        y = targets[:, column]
        if not 0 < y.sum() < len(y):
            continue
        rates = np.array([y[centre == c].mean() for c in centres])
        style.append(roc_auc_score(y, onehot @ rates))
        share.append(rates.max() / (rates.sum() + 1e-9))

    probabilities = np.load("work/oof_shallow.npz", allow_pickle=True)["probabilities"][rows]
    image = [
        roc_auc_score(targets[:, c], probabilities[:, c])
        for c in top
        if 0 < targets[:, c].sum() < len(targets)
    ]

    print(f"\nsentence choice predicted by centre identity : mean AUC {np.mean(style):.3f}")
    print(f"sentence choice predicted from the image     : mean AUC {np.mean(image):.3f}")
    print(f"usage concentrated in one centre             : {np.mean(share):.1%}")
    print("\nthe reporting convention is more predictive than the anatomy, and the")
    print("hidden test centre's convention appears nowhere in the training release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
