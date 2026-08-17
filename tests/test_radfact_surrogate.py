"""The offline RadFact surrogate must never reward the errors RadFact punishes."""

import pytest

from cbct_reasoner.metrics.radfact import (
    LexicalRadFact,
    is_entailed,
    phrase_entailment_score,
    to_phrases,
)

REFERENCE = (
    "The mandibular canal is identifiable bilaterally. "
    "Mucosal thickening is noted in the left maxillary sinus. "
    "No periapical radiolucency is observed."
)


def test_paraphrase_is_entailed() -> None:
    assert (
        phrase_entailment_score(
            "The mandibular canal is identifiable bilaterally.",
            "The mandibular canal is identifiable bilaterally along its course.",
        )
        >= 0.62
    )


def test_opposite_polarity_is_never_entailed() -> None:
    """Asserting a finding the reference explicitly denies is a hallucination."""
    assert (
        phrase_entailment_score(
            "Periapical radiolucency is present at tooth 46.",
            "No periapical radiolucency is observed.",
        )
        == 0.0
    )


def test_wrong_laterality_is_never_entailed() -> None:
    """Side confusion is the error clinicians penalise hardest."""
    assert (
        phrase_entailment_score(
            "Mucosal thickening is noted in the right maxillary sinus.",
            "Mucosal thickening is noted in the left maxillary sinus.",
        )
        == 0.0
    )


def test_bilateral_reference_supports_a_single_side() -> None:
    assert (
        phrase_entailment_score(
            "The left mandibular canal is identifiable.",
            "The mandibular canal is identifiable bilaterally.",
        )
        > 0.0
    )


def test_wrong_tooth_number_is_never_entailed() -> None:
    assert (
        phrase_entailment_score(
            "Tooth 38 is impacted.",
            "Tooth 48 is impacted.",
        )
        == 0.0
    )


def test_precision_and_recall_move_in_the_expected_directions() -> None:
    engine = LexicalRadFact()

    exact = engine.score_case("case", REFERENCE, REFERENCE)
    assert exact.logical_precision == pytest.approx(1.0)
    assert exact.logical_recall == pytest.approx(1.0)

    # Dropping a finding costs recall but not precision.
    partial = engine.score_case(
        "case", "The mandibular canal is identifiable bilaterally.", REFERENCE
    )
    assert partial.logical_precision == pytest.approx(1.0)
    assert partial.logical_recall < 0.5

    # Adding an unsupported finding costs precision but not recall.
    padded = engine.score_case("case", REFERENCE + " Tooth 48 is impacted.", REFERENCE)
    assert padded.logical_recall == pytest.approx(1.0)
    assert padded.logical_precision < 1.0


def test_aggregate_is_f1_of_the_means() -> None:
    engine = LexicalRadFact()
    aggregate, results = engine.score(
        {"a": REFERENCE, "b": "The mandibular canal is identifiable bilaterally."},
        {"a": REFERENCE, "b": REFERENCE},
    )
    mean_precision = sum(r.logical_precision for r in results) / len(results)
    mean_recall = sum(r.logical_recall for r in results) / len(results)
    expected = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)

    assert aggregate.logical_f1 == pytest.approx(expected)


def test_recommendations_are_stripped_before_scoring() -> None:
    """Non-verifiable sentences are dropped, so they cannot dilute precision."""
    phrases = to_phrases(REFERENCE + " Clinical follow-up is recommended.")
    assert all("recommend" not in phrase.lower() for phrase in phrases)


def test_is_entailed_requires_at_least_one_supporting_phrase() -> None:
    assert is_entailed("The mandibular canal is identifiable.", to_phrases(REFERENCE))
    assert not is_entailed("The condyles show erosive change.", to_phrases(REFERENCE))
