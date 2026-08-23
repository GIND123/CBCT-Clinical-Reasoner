"""Evaluation metrics.

``official`` mirrors the grader byte for byte; ``radfact`` supplies the clinical
metric that carries 80% of the final score; ``score`` combines them.

The legacy development proxies (``corpus_bleu4``, ``meteor_lite``,
``clinical_term_f1``) remain exported for the original CLI and tests, but
``bleu_4``/``meteor`` from ``official`` are the numbers that count.
"""

from cbct_reasoner.metrics._legacy import clinical_term_f1
from cbct_reasoner.metrics.official import (
    bleu_4,
    bleu_4_tokenized,
    captioning_metrics,
    meteor,
    meteor_score,
    meteor_score_tokenized,
    tokenize,
)
from cbct_reasoner.metrics.radfact import (
    LexicalRadFact,
    RadFactAggregate,
    RadFactResult,
    phrase_entailment_score,
    radfact_lite_scores,
    to_phrases,
)
from cbct_reasoner.metrics.score import (
    CAPTIONING_WEIGHT,
    CLINICAL_WEIGHT,
    ChallengeScore,
    combine,
    score_reports,
    score_with_radfact_lite,
)

#: Backwards-compatible aliases; both now resolve to the exact grader semantics.
corpus_bleu4 = bleu_4
meteor_lite = meteor_score


def evaluate_pairs(pairs) -> dict[str, float | int]:
    """Score ``(prediction, reference)`` pairs with the official captioning metrics."""
    rows = list(pairs)
    if not rows:
        raise ValueError("At least one prediction/reference pair is required")
    predictions = [row[0] for row in rows]
    references = [row[1] for row in rows]
    result = captioning_metrics(predictions, references)
    result["clinical_term_f1_proxy"] = sum(
        clinical_term_f1(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    ) / len(rows)
    # Historical key names kept so existing notebooks and tests keep working.
    result["bleu_4_proxy"] = result["bleu_4"]
    result["meteor_lite"] = result["meteor"]
    return result


__all__ = [
    "CAPTIONING_WEIGHT",
    "CLINICAL_WEIGHT",
    "ChallengeScore",
    "LexicalRadFact",
    "RadFactAggregate",
    "RadFactResult",
    "bleu_4",
    "bleu_4_tokenized",
    "captioning_metrics",
    "clinical_term_f1",
    "combine",
    "corpus_bleu4",
    "evaluate_pairs",
    "meteor",
    "meteor_lite",
    "meteor_score",
    "meteor_score_tokenized",
    "phrase_entailment_score",
    "radfact_lite_scores",
    "score_reports",
    "score_with_radfact_lite",
    "to_phrases",
    "tokenize",
]
