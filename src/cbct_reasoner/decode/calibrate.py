"""Threshold calibration against the actual ranking objective.

``0.8 * RadFact-F1 + 0.2 * mean(BLEU-4, METEOR)`` is not differentiable, so it is
optimized directly by coordinate ascent over per-prototype thresholds. The whole
approach hinges on making one objective evaluation cheap:

* Entailment between a prototype and a case's reference phrases does not depend
  on the thresholds, so both directions are precomputed once into boolean
  matrices. RadFact then reduces to two array reductions per case.
* METEOR uses the linear-time matcher from ``metrics.official``.
* BLEU reference n-gram counts are precomputed per case.

The result is a full-corpus objective evaluation in well under a second, which
makes a few thousand ascent steps practical. Calibration always runs on
out-of-fold probabilities - fitting thresholds on in-sample predictions produces
a decoder that is confidently wrong on the hidden test set.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cbct_reasoner.decode.decoder import DecoderSettings, ReportDecoder
from cbct_reasoner.metrics.official import (
    BLEU_EPSILON,
    BLEU_WEIGHTS,
    _brevity_penalty,
    build_reference_index,
    meteor_score_fast,
    tokenize,
)
from cbct_reasoner.metrics.radfact import (
    DEFAULT_ENTAILMENT_THRESHOLD,
    phrase_entailment_score,
    to_phrases,
)
from cbct_reasoner.prototypes import PrototypeBank


@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    final: float
    clinical: float
    captioning: float
    logical_precision: float
    logical_recall: float
    bleu_4: float
    meteor: float
    mean_sentences: float

    def to_dict(self) -> dict[str, float]:
        return {
            "final": self.final,
            "clinical": self.clinical,
            "captioning": self.captioning,
            "logical_precision": self.logical_precision,
            "logical_recall": self.logical_recall,
            "bleu_4": self.bleu_4,
            "meteor": self.meteor,
            "mean_sentences": self.mean_sentences,
        }


class CalibrationScorer:
    """Fast, exact-where-it-matters evaluation of a prototype selection."""

    def __init__(
        self,
        bank: PrototypeBank,
        references: Sequence[str],
        *,
        reference_phrases: Sequence[Sequence[str]] | None = None,
        clinical_weight: float = 0.8,
        captioning_weight: float = 0.2,
        entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
    ) -> None:
        if reference_phrases is not None and len(reference_phrases) != len(references):
            raise ValueError("reference_phrases must align with references")
        self.bank = bank
        self.clinical_weight = clinical_weight
        self.captioning_weight = captioning_weight
        self.num_cases = len(references)
        self.num_prototypes = len(bank)

        self.prototype_tokens: list[list[str]] = [tokenize(p.text) for p in bank]
        self.prototype_ngrams: list[list[Counter[tuple[str, ...]]]] = [
            [_ngram_counter(tokens, order) for order in range(1, 5)]
            for tokens in self.prototype_tokens
        ]

        self.reference_tokens: list[list[str]] = [tokenize(text) for text in references]
        self.reference_index = [build_reference_index(tokens) for tokens in self.reference_tokens]
        self.reference_ngrams = [
            [_ngram_counter(tokens, order) for order in range(1, 5)]
            for tokens in self.reference_tokens
        ]
        self.reference_lengths = np.asarray([len(tokens) for tokens in self.reference_tokens])

        phrase_lists = (
            [list(item) for item in reference_phrases]
            if reference_phrases is not None
            else [to_phrases(text) for text in references]
        )
        self.entailed_candidate, self.covers_reference, self.reference_counts = (
            _entailment_matrices(bank, phrase_lists, entailment_threshold)
        )

    # -- objective ---------------------------------------------------------

    def score_selection(self, selections: Sequence[Sequence[int]]) -> ObjectiveBreakdown:
        if len(selections) != self.num_cases:
            raise ValueError(f"expected {self.num_cases} selections, received {len(selections)}")
        precisions = np.zeros(self.num_cases, dtype=np.float64)
        recalls = np.zeros(self.num_cases, dtype=np.float64)
        meteors = np.zeros(self.num_cases, dtype=np.float64)
        lengths = np.zeros(self.num_cases, dtype=np.int64)

        numerators = [0, 0, 0, 0]
        denominators = [0, 0, 0, 0]
        hypothesis_length = 0

        for case in range(self.num_cases):
            chosen = list(selections[case])
            lengths[case] = len(chosen)

            if chosen:
                entailed = int(self.entailed_candidate[case, chosen].sum())
                precisions[case] = entailed / len(chosen)
                covered = self.covers_reference[case][:, chosen].any(axis=1)
                recalls[case] = (
                    float(covered.sum()) / self.reference_counts[case]
                    if self.reference_counts[case]
                    else 0.0
                )

            tokens: list[str] = []
            for index in chosen:
                tokens.extend(self.prototype_tokens[index])
            meteors[case] = meteor_score_fast(
                tokens, self.reference_tokens[case], self.reference_index[case]
            )
            hypothesis_length += len(tokens)
            self._accumulate_bleu(case, chosen, tokens, numerators, denominators)

        precision = float(precisions.mean())
        recall = float(recalls.mean())
        clinical = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        bleu = _finish_bleu(
            numerators, denominators, hypothesis_length, int(self.reference_lengths.sum())
        )
        meteor = float(meteors.mean())
        captioning = (bleu + meteor) / 2.0
        return ObjectiveBreakdown(
            final=self.clinical_weight * clinical + self.captioning_weight * captioning,
            clinical=clinical,
            captioning=captioning,
            logical_precision=precision,
            logical_recall=recall,
            bleu_4=bleu,
            meteor=meteor,
            mean_sentences=float(lengths.mean()),
        )

    def _accumulate_bleu(
        self,
        case: int,
        chosen: Sequence[int],
        tokens: Sequence[str],
        numerators: list[int],
        denominators: list[int],
    ) -> None:
        for order in range(1, 5):
            counts: Counter[tuple[str, ...]] = Counter()
            for index in chosen:
                counts.update(self.prototype_ngrams[index][order - 1])
            # n-grams straddling two concatenated sentences are not captured by
            # the per-prototype counters; add them explicitly so the number
            # matches the grader's tokenization of the joined report.
            counts.update(_boundary_ngrams(self.prototype_tokens, chosen, order))
            reference = self.reference_ngrams[case][order - 1]
            numerators[order - 1] += sum(
                min(count, reference[gram]) for gram, count in counts.items()
            )
            denominators[order - 1] += max(1, max(0, len(tokens) - order + 1))

    # -- helpers -----------------------------------------------------------

    def selections_from(self, probabilities: np.ndarray, decoder: ReportDecoder) -> list[list[int]]:
        return [decoder.select(row) for row in probabilities]


def _ngram_counter(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def _boundary_ngrams(
    prototype_tokens: Sequence[Sequence[str]], chosen: Sequence[int], order: int
) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    if order == 1 or len(chosen) < 2:
        return counts
    for position in range(len(chosen) - 1):
        left = prototype_tokens[chosen[position]]
        right = prototype_tokens[chosen[position + 1]]
        window = list(left[-(order - 1) :]) + list(right[: order - 1])
        for start in range(max(0, len(window) - order + 1)):
            gram = tuple(window[start : start + order])
            if len(gram) == order:
                counts[gram] += 1
    return counts


def _finish_bleu(
    numerators: Sequence[int],
    denominators: Sequence[int],
    hypothesis_length: int,
    reference_length: int,
) -> float:
    if numerators[0] == 0:
        return 0.0
    precisions = [
        (numerator + BLEU_EPSILON) / denominator if numerator == 0 else numerator / denominator
        for numerator, denominator in zip(numerators, denominators, strict=True)
    ]
    return float(
        _brevity_penalty(reference_length, hypothesis_length)
        * math.exp(
            math.fsum(w * math.log(p) for w, p in zip(BLEU_WEIGHTS, precisions, strict=True))
        )
    )


def _entailment_matrices(
    bank: PrototypeBank, phrase_lists: Sequence[Sequence[str]], threshold: float
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Precompute both entailment directions between prototypes and references."""
    num_cases, num_prototypes = len(phrase_lists), len(bank)
    entailed_candidate = np.zeros((num_cases, num_prototypes), dtype=bool)
    covers_reference: list[np.ndarray] = []
    reference_counts = np.zeros(num_cases, dtype=np.int64)

    prototype_texts = bank.texts
    for case, phrases in enumerate(phrase_lists):
        reference_counts[case] = len(phrases)
        coverage = np.zeros((len(phrases), num_prototypes), dtype=bool)
        for index, text in enumerate(prototype_texts):
            for row, phrase in enumerate(phrases):
                # Direction 1: is my sentence supported by the reference?
                if not entailed_candidate[case, index] and (
                    phrase_entailment_score(text, phrase) >= threshold
                ):
                    entailed_candidate[case, index] = True
                # Direction 2: does my sentence cover this reference phrase?
                if phrase_entailment_score(phrase, text) >= threshold:
                    coverage[row, index] = True
        covers_reference.append(coverage)
    return entailed_candidate, covers_reference, reference_counts


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    thresholds: np.ndarray
    objective: ObjectiveBreakdown
    baseline: ObjectiveBreakdown
    rounds: int
    trace: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective.to_dict(),
            "baseline": self.baseline.to_dict(),
            "rounds": self.rounds,
            "trace": list(self.trace),
        }


def calibrate(
    probabilities: np.ndarray,
    scorer: CalibrationScorer,
    bank: PrototypeBank,
    *,
    settings: DecoderSettings | None = None,
    rounds: int = 4,
    global_grid: Sequence[float] | None = None,
    local_grid: Sequence[float] | None = None,
    refine_top: int | None = None,
    initial: np.ndarray | None = None,
    verbose: bool = True,
) -> CalibrationResult:
    """Fit per-prototype thresholds by coordinate ascent on the ranking objective."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (scorer.num_cases, len(bank)):
        raise ValueError(
            f"probabilities must have shape ({scorer.num_cases}, {len(bank)}), "
            f"got {probabilities.shape}"
        )
    settings = settings or DecoderSettings()
    global_grid = list(global_grid or np.round(np.arange(0.05, 0.96, 0.05), 3))
    local_grid = list(local_grid or (0.02, 0.08, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.85, 0.97))

    # One decoder is reused across the whole search; only its threshold vector
    # changes, and the contradiction index is expensive to rebuild.
    decoder = ReportDecoder(bank, np.zeros(len(bank), dtype=np.float32), settings=settings)

    def evaluate(thresholds: np.ndarray) -> ObjectiveBreakdown:
        decoder.thresholds = np.asarray(thresholds, dtype=np.float32)
        return scorer.score_selection(decoder.select_many(probabilities))

    # Stage 1: one shared threshold, to land in the right region cheaply. A warm
    # start is accepted because coordinate ascent is greedy: seeding it with a
    # vector fitted elsewhere can escape a local optimum that the flat start
    # cannot, and it is kept only if it actually scores better.
    best_global, best_score, baseline = 0.5, -1.0, None
    for value in global_grid:
        breakdown = evaluate(np.full(len(bank), value, dtype=np.float32))
        if baseline is None or math.isclose(value, 0.5):
            baseline = breakdown
        if breakdown.final > best_score:
            best_global, best_score = float(value), breakdown.final
    thresholds = np.full(len(bank), best_global, dtype=np.float32)
    current = evaluate(thresholds)

    if initial is not None:
        seeded = np.asarray(initial, dtype=np.float32).reshape(-1)
        if seeded.shape != thresholds.shape:
            raise ValueError(
                f"initial thresholds must have shape {thresholds.shape}, got {seeded.shape}"
            )
        seeded_score = evaluate(seeded)
        if verbose:
            print(
                f"[calibrate] warm start scores {seeded_score.final:.4f} "
                f"vs flat start {current.final:.4f}",
                flush=True,
            )
        if seeded_score.final > current.final:
            thresholds, current = seeded.copy(), seeded_score
    trace = [current.final]
    if verbose:
        print(
            f"[calibrate] global threshold {best_global:.2f} -> final={current.final:.4f} "
            f"(P={current.logical_precision:.3f} R={current.logical_recall:.3f} "
            f"BLEU={current.bleu_4:.4f} METEOR={current.meteor:.4f})",
            flush=True,
        )

    # Stage 2: per-prototype refinement, commonest statements first because they
    # move the objective most and set the precision level later ones compete with.
    # Rare statements are left at the global threshold: with a thousand-statement
    # bank, refining a prototype seen in two cases costs an objective evaluation
    # and cannot move a corpus-level score.
    order = sorted(range(len(bank)), key=lambda index: -bank[index].prevalence)
    if refine_top is not None:
        order = order[:refine_top]
    for round_index in range(rounds):
        improved = 0
        for index in order:
            original = float(thresholds[index])
            best_value, best_breakdown = original, current
            for value in local_grid:
                if math.isclose(value, original):
                    continue
                thresholds[index] = value
                breakdown = evaluate(thresholds)
                if breakdown.final > best_breakdown.final + 1e-9:
                    best_value, best_breakdown = float(value), breakdown
            thresholds[index] = best_value
            if best_value != original:
                improved += 1
                current = best_breakdown
        trace.append(current.final)
        if verbose:
            print(
                f"[calibrate] round {round_index + 1}/{rounds}: {improved} thresholds moved, "
                f"final={current.final:.4f} (sentences={current.mean_sentences:.1f})",
                flush=True,
            )
        if improved == 0:
            break

    assert baseline is not None
    return CalibrationResult(
        thresholds=thresholds,
        objective=current,
        baseline=baseline,
        rounds=len(trace) - 1,
        trace=tuple(trace),
    )


def save_calibration(path: str | Path, result: CalibrationResult) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output
