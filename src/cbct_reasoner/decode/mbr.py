"""Minimum Bayes Risk selection over candidate reports.

When several decoders disagree - prototype selection at different thresholds, a
retrieved neighbour's report, an LLM sample - the right choice is not the most
likely candidate but the one with the highest *expected* metric against the
distribution of plausible references. MBR estimates that expectation using the
other candidates as pseudo-references.

Used for the final ensemble step and for choosing between the two submissions
each team is allowed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cbct_reasoner.metrics.official import (
    bleu_4_tokenized,
    build_reference_index,
    meteor_score_fast,
    tokenize,
)


def utility_matrix(candidates: Sequence[str], *, clinical_weight: float = 0.0) -> np.ndarray:
    """Pairwise captioning utility ``U[i, j]`` = score of candidate i against j."""
    tokenized = [tokenize(candidate) for candidate in candidates]
    indices = [build_reference_index(tokens) for tokens in tokenized]
    size = len(candidates)
    matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i, j] = 1.0
                continue
            meteor = meteor_score_fast(tokenized[i], tokenized[j], indices[j])
            bleu = bleu_4_tokenized([tokenized[i]], [tokenized[j]])
            matrix[i, j] = (meteor + bleu) / 2.0
    if clinical_weight:
        matrix *= 1.0 - clinical_weight
    return matrix


def select(
    candidates: Sequence[str], *, weights: Sequence[float] | None = None
) -> tuple[int, np.ndarray]:
    """Return the index of the MBR-optimal candidate and each candidate's risk-adjusted score."""
    if not candidates:
        raise ValueError("candidates cannot be empty")
    if len(candidates) == 1:
        return 0, np.ones(1, dtype=np.float64)

    matrix = utility_matrix(candidates)
    if weights is None:
        prior = np.ones(len(candidates), dtype=np.float64)
    else:
        prior = np.asarray(weights, dtype=np.float64)
        if prior.shape != (len(candidates),):
            raise ValueError("weights must have one entry per candidate")
    prior = prior / prior.sum()

    # Exclude self-utility so a candidate cannot vote for itself.
    expected = np.zeros(len(candidates), dtype=np.float64)
    for i in range(len(candidates)):
        mask = np.ones(len(candidates), dtype=bool)
        mask[i] = False
        weight = prior[mask]
        expected[i] = float((matrix[i, mask] * weight).sum() / weight.sum())
    return int(np.argmax(expected)), expected
