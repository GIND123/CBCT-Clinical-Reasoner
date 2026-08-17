"""Bit-exact port of the ToothFairy4 grader's captioning metrics.

The organizer's ``evaluate.py`` sets ``RUNNING_ON_GRAND_CHALLENGE`` inside the
platform, which routes both metrics through its *local* implementations:

* BLEU-4  -> ``nltk.translate.bleu_score.corpus_bleu`` with uniform 4-gram
  weights and ``SmoothingFunction().method1``, aggregated over the whole test
  set (corpus level, not per case).
* METEOR  -> ``meteor_lite_score``: exact-token greedy alignment, recall-weighted
  F-mean, cubic chunk penalty. No WordNet, no stemming, no synonyms.

Both are reimplemented here in pure Python so the inference container needs no
NLTK, and ``tests/test_official_metrics.py`` asserts agreement with NLTK to
machine precision. Optimizing against these functions therefore optimizes the
leaderboard directly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

#: The grader's tokenizer, character for character.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

#: ``SmoothingFunction`` default; NLTK adds this to zero-count numerators.
BLEU_EPSILON = 0.1
BLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


def tokenize(text: str) -> list[str]:
    """Reproduce ``evaluate.py::tokenize`` exactly."""
    return [token for token in _TOKEN_RE.findall(text.lower()) if token.strip()]


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def _modified_precision(
    reference_tokens: Sequence[str], hypothesis_tokens: Sequence[str], order: int
) -> tuple[int, int]:
    """NLTK ``modified_precision`` for the single-reference case.

    Returns the unnormalized ``(numerator, denominator)`` pair; NLTK accumulates
    these across the corpus before dividing, which is why a per-case average of
    BLEU is *not* the same number.
    """
    hypothesis_counts = _ngrams(hypothesis_tokens, order)
    reference_counts = _ngrams(reference_tokens, order)
    numerator = sum(min(count, reference_counts[gram]) for gram, count in hypothesis_counts.items())
    denominator = max(1, sum(hypothesis_counts.values()))
    return numerator, denominator


def _brevity_penalty(reference_length: int, hypothesis_length: int) -> float:
    if hypothesis_length > reference_length:
        return 1.0
    if hypothesis_length == 0:
        return 0.0
    return math.exp(1.0 - reference_length / hypothesis_length)


def bleu_4(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Corpus-level BLEU-4 matching NLTK ``corpus_bleu`` + ``method1`` smoothing."""
    prediction_tokens = [tokenize(item) for item in predictions]
    reference_tokens = [tokenize(item) for item in references]
    if len(prediction_tokens) != len(reference_tokens):
        raise ValueError("predictions and references must have the same length")
    return bleu_4_tokenized(prediction_tokens, reference_tokens)


def bleu_4_tokenized(
    prediction_tokens: Sequence[Sequence[str]], reference_tokens: Sequence[Sequence[str]]
) -> float:
    """BLEU-4 over pre-tokenized text; the hot path during threshold search."""
    if not prediction_tokens:
        return 0.0

    numerators = [0, 0, 0, 0]
    denominators = [0, 0, 0, 0]
    hypothesis_length = 0
    reference_length = 0

    for hypothesis, reference in zip(prediction_tokens, reference_tokens, strict=True):
        for order in range(1, 5):
            numerator, denominator = _modified_precision(reference, hypothesis, order)
            numerators[order - 1] += numerator
            denominators[order - 1] += denominator
        hypothesis_length += len(hypothesis)
        reference_length += len(reference)

    # NLTK short-circuits when no unigram matches at all.
    if numerators[0] == 0:
        return 0.0

    precisions = [
        (numerator + BLEU_EPSILON) / denominator if numerator == 0 else numerator / denominator
        for numerator, denominator in zip(numerators, denominators, strict=True)
    ]
    score = _brevity_penalty(reference_length, hypothesis_length) * math.exp(
        math.fsum(weight * math.log(p) for weight, p in zip(BLEU_WEIGHTS, precisions, strict=True))
    )
    return float(score)


def _greedy_match_indices(
    prediction_tokens: Sequence[str], reference_tokens: Sequence[str]
) -> list[int]:
    """First-fit greedy alignment, identical to the grader's implementation."""
    used = [False] * len(reference_tokens)
    indices: list[int] = []
    for token in prediction_tokens:
        for index, reference_token in enumerate(reference_tokens):
            if used[index] or token != reference_token:
                continue
            used[index] = True
            indices.append(index)
            break
    return indices


def _chunk_count(indices: Sequence[int]) -> int:
    if not indices:
        return 0
    return 1 + sum(
        1
        for current, previous in zip(indices[1:], indices[:-1], strict=True)
        if current != previous + 1
    )


def meteor_score_tokenized(
    prediction_tokens: Sequence[str], reference_tokens: Sequence[str]
) -> float:
    """``meteor_lite_score``: recall-weighted F-mean with a cubic fragmentation penalty.

    ``F = 10PR / (R + 9P)`` weights recall roughly nine times precision, so
    coverage of the reference beats terseness. The penalty
    ``0.5 * (chunks/matches)^3`` then punishes scattered single-token hits, which
    is why reusing *contiguous* reference phrasing scores far above reusing the
    same words in a different order.
    """
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    matched = _greedy_match_indices(prediction_tokens, reference_tokens)
    matches = len(matched)
    if matches == 0:
        return 0.0

    precision = matches / len(prediction_tokens)
    recall = matches / len(reference_tokens)
    denominator = recall + 9.0 * precision
    if denominator == 0.0:
        return 0.0

    f_mean = (10.0 * precision * recall) / denominator
    penalty = 0.5 * (_chunk_count(matched) / matches) ** 3
    return float((1.0 - penalty) * f_mean)


def meteor_score(prediction: str, reference: str) -> float:
    return meteor_score_tokenized(tokenize(prediction), tokenize(reference))


def build_reference_index(reference_tokens: Sequence[str]) -> dict[str, list[int]]:
    """Positions of each token value in the reference, ascending.

    Precomputed once per case so calibration can re-score thousands of candidate
    reports against the same reference without rescanning it.
    """
    index: dict[str, list[int]] = {}
    for position, token in enumerate(reference_tokens):
        index.setdefault(token, []).append(position)
    return index


def meteor_score_fast(
    prediction_tokens: Sequence[str],
    reference_tokens: Sequence[str],
    reference_index: dict[str, list[int]] | None = None,
) -> float:
    """``meteor_score_tokenized`` in O(P + R) instead of O(P * R).

    The grader's greedy matcher assigns each prediction token to the *earliest
    still-unused* occurrence of that value in the reference. Walking a per-value
    cursor reproduces that assignment exactly - verified against the reference
    implementation in ``tests/test_official_metrics.py`` - and turns threshold
    search from hours into seconds.
    """
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    index = (
        reference_index if reference_index is not None else build_reference_index(reference_tokens)
    )
    cursors: dict[str, int] = {}
    matched: list[int] = []
    for token in prediction_tokens:
        positions = index.get(token)
        if positions is None:
            continue
        cursor = cursors.get(token, 0)
        if cursor >= len(positions):
            continue
        matched.append(positions[cursor])
        cursors[token] = cursor + 1

    matches = len(matched)
    if matches == 0:
        return 0.0

    precision = matches / len(prediction_tokens)
    recall = matches / len(reference_tokens)
    denominator = recall + 9.0 * precision
    if denominator == 0.0:
        return 0.0

    f_mean = (10.0 * precision * recall) / denominator
    # `matched` is produced in prediction order but the grader counts chunks over
    # the same sequence, so no sort is applied here either.
    penalty = 0.5 * (_chunk_count(matched) / matches) ** 3
    return float((1.0 - penalty) * f_mean)


def meteor(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Mean per-case METEOR, matching ``meteor_lite_batch``."""
    prediction_list = list(predictions)
    reference_list = list(references)
    if len(prediction_list) != len(reference_list):
        raise ValueError("predictions and references must have the same length")
    if not prediction_list:
        return 0.0
    scores = [
        meteor_score(prediction, reference)
        for prediction, reference in zip(prediction_list, reference_list, strict=True)
    ]
    return float(sum(scores) / len(scores))


def captioning_metrics(predictions: Iterable[str], references: Iterable[str]) -> dict[str, float]:
    """Both public metrics plus the challenge's averaged captioning score."""
    prediction_list = list(predictions)
    reference_list = list(references)
    bleu = bleu_4(prediction_list, reference_list)
    meteor_value = meteor(prediction_list, reference_list)
    return {
        "bleu_4": bleu,
        "meteor": meteor_value,
        "captioning": (bleu + meteor_value) / 2.0,
        "num_cases": len(prediction_list),
    }
