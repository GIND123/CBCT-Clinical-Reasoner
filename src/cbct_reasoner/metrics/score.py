"""The ODIN 2026 Phase-1 ranking objective.

``Final = 0.8 * RadFact-F1 + 0.2 * mean(BLEU-4, METEOR)``

Every decoder decision in this repository is made against ``final_score``
computed on out-of-fold predictions, so local improvements are leaderboard
improvements rather than proxy improvements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from cbct_reasoner.metrics.official import captioning_metrics
from cbct_reasoner.metrics.radfact import LexicalRadFact, RadFactAggregate

CLINICAL_WEIGHT = 0.8
CAPTIONING_WEIGHT = 0.2


@dataclass(frozen=True, slots=True)
class ChallengeScore:
    final: float
    clinical: float
    captioning: float
    logical_precision: float
    logical_recall: float
    bleu_4: float
    meteor: float
    num_cases: int
    clinical_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def combine(
    *,
    clinical: float,
    bleu_4: float,
    meteor: float,
    clinical_weight: float = CLINICAL_WEIGHT,
    captioning_weight: float = CAPTIONING_WEIGHT,
) -> float:
    captioning = (bleu_4 + meteor) / 2.0
    return clinical_weight * clinical + captioning_weight * captioning


def score_reports(
    predictions_by_id: Mapping[str, str],
    references_by_id: Mapping[str, str | Sequence[str]],
    *,
    radfact: LexicalRadFact | None = None,
    clinical_weight: float = CLINICAL_WEIGHT,
    captioning_weight: float = CAPTIONING_WEIGHT,
    clinical_source: str = "lexical-surrogate",
) -> ChallengeScore:
    """Score a prediction set with the surrogate clinical metric.

    ``references_by_id`` may map to several reference reports for one case; the
    captioning metrics use the first (the consensus reference), while the
    clinical surrogate scores against the union so that a finding documented by
    any annotator counts as supported.
    """
    case_ids = sorted(set(predictions_by_id) & set(references_by_id))
    if not case_ids:
        raise ValueError("No overlapping case identifiers between predictions and references")

    predictions = [predictions_by_id[case_id] for case_id in case_ids]
    primary_references = [_primary(references_by_id[case_id]) for case_id in case_ids]

    caption = captioning_metrics(predictions, primary_references)
    engine = radfact or LexicalRadFact()
    aggregate, _ = engine.score(
        {case_id: predictions_by_id[case_id] for case_id in case_ids},
        {case_id: _joined(references_by_id[case_id]) for case_id in case_ids},
    )
    return _assemble(
        aggregate=aggregate,
        caption=caption,
        clinical_weight=clinical_weight,
        captioning_weight=captioning_weight,
        clinical_source=clinical_source,
    )


def score_with_radfact_lite(
    predictions_by_id: Mapping[str, str],
    references_by_id: Mapping[str, str | Sequence[str]],
    **radfact_kwargs: Any,
) -> ChallengeScore:
    """Score using the organizer's real LLM-based RadFact implementation."""
    from cbct_reasoner.metrics.radfact import RadFactAggregate as _Aggregate
    from cbct_reasoner.metrics.radfact import radfact_lite_scores

    case_ids = sorted(set(predictions_by_id) & set(references_by_id))
    if not case_ids:
        raise ValueError("No overlapping case identifiers between predictions and references")

    predictions = [predictions_by_id[case_id] for case_id in case_ids]
    primary_references = [_primary(references_by_id[case_id]) for case_id in case_ids]
    caption = captioning_metrics(predictions, primary_references)

    payload = radfact_lite_scores(
        {case_id: predictions_by_id[case_id] for case_id in case_ids},
        {case_id: _primary(references_by_id[case_id]) for case_id in case_ids},
        **radfact_kwargs,
    )
    aggregates = payload["aggregates"]
    assert isinstance(aggregates, dict)
    aggregate = _Aggregate(
        logical_precision=float(aggregates["logical_precision"]),
        logical_recall=float(aggregates["logical_recall"]),
        logical_f1=float(aggregates["logical_f1"]),
        num_samples=int(aggregates["num_samples"]),
    )
    return _assemble(
        aggregate=aggregate,
        caption=caption,
        clinical_weight=CLINICAL_WEIGHT,
        captioning_weight=CAPTIONING_WEIGHT,
        clinical_source="radfact-lite",
    )


def _assemble(
    *,
    aggregate: RadFactAggregate,
    caption: dict[str, float],
    clinical_weight: float,
    captioning_weight: float,
    clinical_source: str,
) -> ChallengeScore:
    return ChallengeScore(
        final=combine(
            clinical=aggregate.logical_f1,
            bleu_4=caption["bleu_4"],
            meteor=caption["meteor"],
            clinical_weight=clinical_weight,
            captioning_weight=captioning_weight,
        ),
        clinical=aggregate.logical_f1,
        captioning=caption["captioning"],
        logical_precision=aggregate.logical_precision,
        logical_recall=aggregate.logical_recall,
        bleu_4=caption["bleu_4"],
        meteor=caption["meteor"],
        num_cases=int(caption["num_cases"]),
        clinical_source=clinical_source,
    )


def _primary(reference: str | Sequence[str]) -> str:
    return reference if isinstance(reference, str) else reference[0]


def _joined(reference: str | Sequence[str]) -> str:
    return reference if isinstance(reference, str) else " ".join(reference)
