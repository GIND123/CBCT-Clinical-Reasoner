"""RadFact logical precision / recall - the 80% weight in the final score.

Two implementations are provided.

``LexicalRadFact``
    An offline, deterministic surrogate. Threshold calibration evaluates the
    objective tens of thousands of times, so an LLM-in-the-loop metric is not
    usable there. This surrogate scores phrase-level entailment from ontology
    concept agreement, polarity, laterality, tooth numbers, and token
    containment - the same signals the grader's LLM keys on.

``radfact_lite_scores``
    A bridge to the organizer's actual ``radfact_lite`` package. Point it at
    OpenAI, a local Ollama server, or any OpenAI-compatible endpoint (vLLM on
    Modal) to obtain the real number for final model selection.

The surrogate is for *ranking* decoder variants cheaply; ``radfact_lite`` is for
*trusting* the result. Never report a surrogate score as the challenge metric.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from cbct_reasoner.metrics.official import tokenize
from cbct_reasoner.ontology import extract_mentions
from cbct_reasoner.text import is_verifiable, split_phrases

#: Above this surrogate similarity a candidate phrase counts as entailed.
DEFAULT_ENTAILMENT_THRESHOLD = 0.62

_STOPWORDS = frozenset(
    """
    a an and are as at be been both by for from has have in is it its of on or
    that the there these this to was were with within without no not
    """.split()
)


@dataclass(frozen=True, slots=True)
class RadFactResult:
    """Mirrors ``radfact_lite.RadFactSampleResult`` field for field."""

    sample_id: str
    logical_precision: float
    logical_recall: float
    logical_f1: float
    entailed_candidate_count: int
    entailed_reference_count: int
    candidate_count: int
    reference_count: int


@dataclass(frozen=True, slots=True)
class RadFactAggregate:
    logical_precision: float
    logical_recall: float
    logical_f1: float
    num_samples: int


def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def _divide_or_zero(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


@lru_cache(maxsize=100_000)
def _phrase_signature(
    phrase: str,
) -> tuple[frozenset[str], frozenset[tuple[str, str]], str, frozenset[int]]:
    """Cache the expensive parse of a phrase into comparable parts.

    Prototype sentences repeat across thousands of candidate reports during
    calibration, so memoizing this turns the surrogate from minutes to seconds.
    """
    content = frozenset(
        token for token in tokenize(phrase) if token.isalnum() and token not in _STOPWORDS
    )
    mentions = extract_mentions(phrase)
    concepts = frozenset((mention.concept, mention.polarity) for mention in mentions)
    laterality = next(
        (m.laterality for m in mentions if m.laterality != "unspecified"), "unspecified"
    )
    teeth = frozenset(tooth for mention in mentions for tooth in mention.teeth)
    return content, concepts, laterality, teeth


def phrase_entailment_score(candidate: str, reference: str) -> float:
    """Surrogate probability that ``reference`` entails ``candidate``.

    Weighted so that asserting the *opposite polarity* of a reference finding, or
    swapping laterality, cannot be scored as entailment - those are exactly the
    hallucinations RadFact precision is designed to punish.
    """
    cand_tokens, cand_concepts, cand_side, cand_teeth = _phrase_signature(candidate)
    ref_tokens, ref_concepts, ref_side, ref_teeth = _phrase_signature(reference)

    if cand_concepts and ref_concepts:
        cand_keys = {key for key, _ in cand_concepts}
        ref_keys = {key for key, _ in ref_concepts}
        shared = cand_keys & ref_keys
        if not shared:
            return 0.0
        # A concept asserted with contradictory polarity is never entailed.
        for key in shared:
            cand_polarity = {p for k, p in cand_concepts if k == key}
            ref_polarity = {p for k, p in ref_concepts if k == key}
            if {"present"} <= cand_polarity and {"absent"} == ref_polarity:
                return 0.0
            if {"absent"} <= cand_polarity and {"present"} == ref_polarity:
                return 0.0
        concept_score = len(shared) / len(cand_keys)
    elif cand_concepts or ref_concepts:
        concept_score = 0.0
    else:
        concept_score = 0.5

    # A bilateral reference does support a single-sided claim; anything else
    # that disagrees on side is a laterality error, never an entailment.
    if (
        cand_side != "unspecified"
        and ref_side != "unspecified"
        and cand_side != ref_side
        and not (ref_side == "bilateral" and cand_side in {"left", "right"})
    ):
        return 0.0

    if cand_teeth and ref_teeth and not (cand_teeth & ref_teeth):
        return 0.0

    containment = _divide_or_zero(len(cand_tokens & ref_tokens), len(cand_tokens))
    tooth_bonus = 0.1 if cand_teeth and (cand_teeth & ref_teeth) else 0.0
    return min(1.0, 0.55 * concept_score + 0.45 * containment + tooth_bonus)


def is_entailed(
    phrase: str, evidence: Sequence[str], *, threshold: float = DEFAULT_ENTAILMENT_THRESHOLD
) -> bool:
    return any(phrase_entailment_score(phrase, item) >= threshold for item in evidence)


def to_phrases(report: str | Sequence[str], *, drop_non_verifiable: bool = True) -> list[str]:
    """Parse a narrative report the way RadFact's phrase extractor would."""
    phrases = list(report) if not isinstance(report, str) else split_phrases(report)
    if drop_non_verifiable:
        phrases = [phrase for phrase in phrases if is_verifiable(phrase)]
    return phrases


class LexicalRadFact:
    """Offline RadFact surrogate used for decoder calibration and ablations."""

    def __init__(self, *, threshold: float = DEFAULT_ENTAILMENT_THRESHOLD) -> None:
        self.threshold = threshold

    def score_case(
        self, sample_id: str, candidate: str | Sequence[str], reference: str | Sequence[str]
    ) -> RadFactResult:
        candidate_phrases = to_phrases(candidate)
        reference_phrases = to_phrases(reference)
        entailed_candidate = sum(
            is_entailed(phrase, reference_phrases, threshold=self.threshold)
            for phrase in candidate_phrases
        )
        entailed_reference = sum(
            is_entailed(phrase, candidate_phrases, threshold=self.threshold)
            for phrase in reference_phrases
        )
        precision = _divide_or_zero(entailed_candidate, len(candidate_phrases))
        recall = _divide_or_zero(entailed_reference, len(reference_phrases))
        return RadFactResult(
            sample_id=sample_id,
            logical_precision=precision,
            logical_recall=recall,
            logical_f1=_f1(precision, recall),
            entailed_candidate_count=entailed_candidate,
            entailed_reference_count=entailed_reference,
            candidate_count=len(candidate_phrases),
            reference_count=len(reference_phrases),
        )

    def score(
        self,
        candidates_by_id: Mapping[str, str | Sequence[str]],
        references_by_id: Mapping[str, str | Sequence[str]],
    ) -> tuple[RadFactAggregate, list[RadFactResult]]:
        common = sorted(set(candidates_by_id) & set(references_by_id))
        results = [
            self.score_case(sample_id, candidates_by_id[sample_id], references_by_id[sample_id])
            for sample_id in common
        ]
        return aggregate(results), results


def aggregate(results: Iterable[RadFactResult]) -> RadFactAggregate:
    """Harmonic mean of the *mean* precision and recall, as the grader computes it."""
    items = list(results)
    if not items:
        return RadFactAggregate(0.0, 0.0, 0.0, 0)
    mean_precision = sum(item.logical_precision for item in items) / len(items)
    mean_recall = sum(item.logical_recall for item in items) / len(items)
    return RadFactAggregate(
        logical_precision=mean_precision,
        logical_recall=mean_recall,
        logical_f1=_f1(mean_precision, mean_recall),
        num_samples=len(items),
    )


def radfact_lite_scores(
    candidates_by_id: Mapping[str, str],
    references_by_id: Mapping[str, str],
    *,
    model: str | None = None,
    provider: str = "openai",
    base_url: str | None = None,
    api_key_env_var: str = "OPENAI_API_KEY",
    timeout: float = 60.0,
    max_retries: int = 2,
) -> dict[str, object]:
    """Run the organizer's real ``radfact_lite`` pipeline.

    ``provider="ollama"`` (or ``base_url`` pointing at a vLLM server) keeps the
    clinical reports off third-party APIs, which matters because these are
    patient reports. Install with ``pip install '.[radfact]'``.
    """
    from dataclasses import asdict

    from radfact_lite import ModelConfig, PipelineModels, RadFactLitePipeline, ReportType

    config = ModelConfig(
        model=model or ("llama3.1" if provider == "ollama" else "gpt-4o-mini"),
        provider=provider,  # type: ignore[arg-type]
        api_key_env_var=api_key_env_var,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    pipeline = RadFactLitePipeline(
        models=PipelineModels(parse_model=config, entailment_model=config, filtering_model=config),
        report_type=ReportType.TOOTHFAIRY,
    )
    aggregate_result, per_case = pipeline.compute_radfact(
        candidates_by_id=dict(candidates_by_id),
        references_by_id=dict(references_by_id),
        is_narrative_text=True,
        # The grader passes False: negative and normal statements are scored too,
        # which is why confidently asserting common normals raises both P and R.
        filter_negatives=False,
    )
    return {
        "aggregates": asdict(aggregate_result),
        "results": [asdict(item) for item in per_case],
    }
