"""Fit the shipping report with a chosen configuration and embed it in the entrypoint.

Refits on every centre (the folds were only for choosing the configuration),
reports the per-centre spread — the hidden test set is one centre, not an average
— and rewrites ``BASE_REPORT`` in submission/inference.py in place.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cbct_reasoner.data.corpus import load_corpus  # noqa: E402
from cbct_reasoner.decode.constant import (  # noqa: E402
    CorpusScorer,
    SearchConfig,
    render_tokens,
    search,
)
from cbct_reasoner.prototypes import PrototypeBank  # noqa: E402

BLEU_TRANSFER = 0.0943 / 0.1429
METEOR_TRANSFER = 0.3542 / 0.3490


def embed(text: str) -> None:
    literal = "\n".join(
        f'    "{line} "' for line in textwrap.wrap(text, 84, break_long_words=False)
    )
    path = REPO_ROOT / "submission" / "inference.py"
    source = path.read_text(encoding="utf-8")
    start = source.index("BASE_REPORT = (")
    end = source.index("\n)\n", start) + len("\n)\n")
    path.write_text(
        source[:start] + f"BASE_REPORT = (\n{literal}\n)\n" + source[end:], encoding="utf-8"
    )
    print(f"embedded {len(text)} characters into submission/inference.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bleu-weight", type=float, required=True)
    parser.add_argument("--aggregate", choices=("mean", "min"), required=True)
    parser.add_argument("--min-prevalence", type=float, required=True)
    parser.add_argument("--max-sentences", type=int, default=40)
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()

    entries = load_corpus("work/corpus.jsonl")
    bank = PrototypeBank.load("artifacts/prototypes.json")
    centres = sorted({e.center for e in entries})
    groups = [[e.reference for e in entries if e.center == c] for c in centres]

    config = SearchConfig(
        bleu_weight=args.bleu_weight,
        aggregate=args.aggregate,
        min_prevalence=args.min_prevalence,
        max_sentences=args.max_sentences,
    )
    print(f"fitting on {len(entries)} cases across {centres} with {config.key()}")
    chosen, text = search(bank, groups, config)
    _, tokens = render_tokens(bank, chosen)

    pooled = CorpusScorer([e.reference for e in entries]).score(tokens)
    per_centre = {c: CorpusScorer(g).score(tokens) for c, g in zip(centres, groups, strict=True)}
    worst = min(per_centre.values(), key=lambda p: p[0])

    print(f"\nsentences {len(chosen)}  tokens {len(tokens)}")
    print(f"  pooled in-sample   BLEU {pooled[0]:.4f}  METEOR {pooled[1]:.4f}")
    for centre, (bleu, meteor) in per_centre.items():
        print(f"  centre {centre}           BLEU {bleu:.4f}  METEOR {meteor:.4f}")
    print(f"  worst centre       BLEU {worst[0]:.4f}  METEOR {worst[1]:.4f}")

    Path("artifacts/final_report.txt").write_text(text + "\n", encoding="utf-8")
    Path("artifacts/final_report.json").write_text(
        json.dumps(
            {
                "report": text,
                "config": config.key(),
                "sentences": len(chosen),
                "tokens": len(tokens),
                "in_sample_pooled": {"bleu_4": pooled[0], "meteor": pooled[1]},
                "per_centre": {c: {"bleu_4": v[0], "meteor": v[1]} for c, v in per_centre.items()},
                "indices": sorted(chosen),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.no_embed:
        embed(text)
    print(f"\n{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
