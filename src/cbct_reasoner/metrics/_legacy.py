from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CLINICAL_TERMS = {
    "implant": re.compile(r"\bimplants?\b", re.I),
    "sinus": re.compile(r"\b(?:maxillary\s+)?sinus(?:es)?\b", re.I),
    "canal": re.compile(r"\b(?:mandibular|alveolar)\s+canal\b", re.I),
    "bone": re.compile(r"\bbone\b", re.I),
    "lesion": re.compile(r"\b(?:lesion|radiolucenc\w*|cyst\w*)\b", re.I),
    "impaction": re.compile(r"\b(?:impacted|impaction)\b", re.I),
    "edentulous": re.compile(r"\bedentul\w*\b", re.I),
}


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.casefold()) if token.strip()]


def corpus_bleu4(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Dependency-free corpus BLEU-4 with add-one smoothing.

    This is a fast development proxy. Official challenge scores must come from the
    organizer evaluator because tokenization and package versions can affect results.
    """
    pairs = list(zip(predictions, references, strict=True))
    if not pairs:
        return 0.0
    clipped = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    prediction_length = 0
    reference_length = 0
    for prediction, reference in pairs:
        candidate = tokenize(prediction)
        truth = tokenize(reference)
        prediction_length += len(candidate)
        reference_length += len(truth)
        for order in range(1, 5):
            candidate_counts = Counter(_ngrams(candidate, order))
            reference_counts = Counter(_ngrams(truth, order))
            clipped[order - 1] += sum(
                min(count, reference_counts[gram]) for gram, count in candidate_counts.items()
            )
            totals[order - 1] += sum(candidate_counts.values())
    if prediction_length == 0:
        return 0.0
    precisions = [(match + 1) / (total + 1) for match, total in zip(clipped, totals, strict=True)]
    brevity = (
        1.0
        if prediction_length > reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return float(brevity * math.exp(sum(math.log(value) for value in precisions) / 4))


def meteor_lite(prediction: str, reference: str) -> float:
    """Exact-token METEOR-style proxy matching the organizer fallback structure."""
    candidate = tokenize(prediction)
    truth = tokenize(reference)
    if not candidate and not truth:
        return 1.0
    if not candidate or not truth:
        return 0.0
    used = [False] * len(truth)
    indices: list[int] = []
    for token in candidate:
        for index, reference_token in enumerate(truth):
            if not used[index] and token == reference_token:
                used[index] = True
                indices.append(index)
                break
    matches = len(indices)
    if not matches:
        return 0.0
    precision = matches / len(candidate)
    recall = matches / len(truth)
    f_mean = (10 * precision * recall) / (recall + 9 * precision)
    chunks = 1 + sum(
        current != previous + 1 for previous, current in zip(indices, indices[1:], strict=False)
    )
    penalty = 0.5 * (chunks / matches) ** 3
    return float((1 - penalty) * f_mean)


def clinical_term_f1(prediction: str, reference: str) -> float:
    """Small, interpretable terminology-overlap diagnostic; not an official metric."""
    candidate = {name for name, pattern in CLINICAL_TERMS.items() if pattern.search(prediction)}
    truth = {name for name, pattern in CLINICAL_TERMS.items() if pattern.search(reference)}
    if not candidate and not truth:
        return 1.0
    if not candidate or not truth:
        return 0.0
    overlap = len(candidate & truth)
    precision = overlap / len(candidate)
    recall = overlap / len(truth)
    return float(2 * precision * recall / (precision + recall)) if overlap else 0.0


def evaluate_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, float | int]:
    rows = list(pairs)
    predictions = [row[0] for row in rows]
    references = [row[1] for row in rows]
    if not rows:
        raise ValueError("At least one prediction/reference pair is required")
    return {
        "bleu_4_proxy": corpus_bleu4(predictions, references),
        "meteor_lite": sum(
            meteor_lite(prediction, reference)
            for prediction, reference in zip(predictions, references, strict=True)
        )
        / len(rows),
        "clinical_term_f1_proxy": sum(
            clinical_term_f1(prediction, reference)
            for prediction, reference in zip(predictions, references, strict=True)
        )
        / len(rows),
        "num_cases": len(rows),
    }


def _ngrams(tokens: list[str], order: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]
