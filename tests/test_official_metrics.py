"""Parity tests against the organizer's evaluator.

``evaluation/evaluate.py`` on Grand Challenge routes both captioning metrics
through its local implementations: NLTK ``corpus_bleu`` with ``method1``
smoothing, and its own exact-match ``meteor_lite_score``. If these tests ever
fail, local calibration has stopped predicting the leaderboard.
"""

from __future__ import annotations

import random

import pytest

from cbct_reasoner.metrics.official import (
    bleu_4,
    build_reference_index,
    meteor,
    meteor_score_fast,
    meteor_score_tokenized,
    tokenize,
)

nltk = pytest.importorskip("nltk", reason="nltk is only needed for grader parity checks")

VOCAB = (
    "the mandibular canal is bilaterally visible maxillary sinus mucosal thickening "
    "tooth 48 impacted no periapical lesion alveolar bone height adequate implant . ,"
).split()  # noqa: SIM905 - a readable sentence beats a 23-element literal


def nltk_bleu(predictions: list[str], references: list[str]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    return float(
        corpus_bleu(
            list_of_references=[[tokenize(reference)] for reference in references],
            hypotheses=[tokenize(prediction) for prediction in predictions],
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def random_corpus(seed: int, size: int = 30) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    predictions, references = [], []
    for _ in range(size):
        references.append(" ".join(rng.choice(VOCAB) for _ in range(rng.randint(3, 60))))
        predictions.append(" ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 70))))
    return predictions, references


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_corpus_bleu_matches_nltk(seed: int) -> None:
    predictions, references = random_corpus(seed)
    assert bleu_4(predictions, references) == pytest.approx(
        nltk_bleu(predictions, references), abs=1e-12
    )


@pytest.mark.parametrize("seed", [4, 5])
def test_per_case_bleu_matches_nltk(seed: int) -> None:
    predictions, references = random_corpus(seed, size=12)
    for prediction, reference in zip(predictions, references, strict=True):
        assert bleu_4([prediction], [reference]) == pytest.approx(
            nltk_bleu([prediction], [reference]), abs=1e-12
        )


def test_bleu_edge_cases() -> None:
    text = "the mandibular canal is clearly visible bilaterally"
    assert bleu_4([text], [text]) == pytest.approx(1.0)
    assert bleu_4([""], [text]) == 0.0
    assert bleu_4(["zzz qqq"], [text]) == 0.0
    assert bleu_4([], []) == 0.0


@pytest.mark.parametrize("seed", range(6, 12))
def test_fast_meteor_equals_reference_implementation(seed: int) -> None:
    """The O(P+R) matcher must reproduce the grader's O(P*R) greedy alignment."""
    rng = random.Random(seed)
    for _ in range(60):
        reference = [rng.choice(VOCAB) for _ in range(rng.randint(0, 90))]
        prediction = [rng.choice(VOCAB) for _ in range(rng.randint(0, 110))]
        expected = meteor_score_tokenized(prediction, reference)
        assert meteor_score_fast(prediction, reference) == pytest.approx(expected, abs=1e-12)
        assert meteor_score_fast(
            prediction, reference, build_reference_index(reference)
        ) == pytest.approx(expected, abs=1e-12)


def test_meteor_is_recall_weighted() -> None:
    """F = 10PR/(R+9P) weights recall about nine times precision.

    Covering the reference verbosely beats being terse and precise, which is the
    single most important property to exploit when choosing report length.
    """
    reference = "alpha beta gamma delta epsilon zeta eta theta"
    high_recall = reference + " unrelated filler words appended here"
    high_precision = "alpha beta"

    assert meteor([high_recall], [reference]) > meteor([high_precision], [reference])


def test_meteor_rewards_contiguous_reuse() -> None:
    """The cubic chunk penalty punishes scattered matches over ordered ones."""
    reference = "alpha beta gamma delta epsilon zeta"
    contiguous = "alpha beta gamma delta epsilon zeta"
    scrambled = "zeta delta beta alpha epsilon gamma"

    assert meteor([contiguous], [reference]) > meteor([scrambled], [reference])


def test_empty_prediction_scores_zero() -> None:
    """A missing result is scored as an empty report, i.e. zero on everything."""
    reference = "the mandibular canal is visible bilaterally along its course"
    assert meteor([""], [reference]) == 0.0
    assert bleu_4([""], [reference]) == 0.0
