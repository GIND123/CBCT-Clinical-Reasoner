"""The deployable decoder: finding probabilities in, narrative report out.

Selection is a per-prototype threshold decision, not a top-k cut. That is a
direct consequence of how RadFact scores a report:

* Precision divides by the number of phrases *you* emit, so an extra sentence
  helps only when its entailment probability exceeds the precision you already
  have.
* Recall divides by the number of reference phrases, so a sentence that covers a
  commonly-reported finding pays even when you are unsure.

Those two pressures balance at a different point for every statement - a
near-universal normal is worth emitting at low confidence, a rare specific
finding is not - so each prototype gets its own threshold, fitted on out-of-fold
probabilities in ``calibrate.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cbct_reasoner.ontology import extract_mentions
from cbct_reasoner.prototypes import PrototypeBank
from cbct_reasoner.text import join_report

DECODER_VERSION = 2


def prototype_polarities(bank: PrototypeBank) -> list[dict[str, str]]:
    """``{concept: polarity}`` for each prototype, derived from its text."""
    profiles: list[dict[str, str]] = []
    for prototype in bank:
        profile: dict[str, str] = {}
        for mention in extract_mentions(prototype.text):
            profile[mention.concept] = mention.polarity
        profiles.append(profile)
    return profiles


def contradiction_pairs(bank: PrototypeBank) -> set[tuple[int, int]]:
    """Prototype pairs that assert opposite polarity on a shared concept.

    Emitting both halves of a contradiction is doubly costly: one of the two
    phrases is guaranteed not to be entailed, and a self-contradicting report is
    exactly what a maxillofacial surgeon penalises in the Phase-2 arena.
    """
    profiles = prototype_polarities(bank)
    pairs: set[tuple[int, int]] = set()
    for left in range(len(bank)):
        for right in range(left + 1, len(bank)):
            shared = set(profiles[left]) & set(profiles[right])
            for concept in shared:
                polarities = {profiles[left][concept], profiles[right][concept]}
                if polarities == {"present", "absent"}:
                    pairs.add((left, right))
                    break
    return pairs


@dataclass(frozen=True, slots=True)
class DecoderSettings:
    min_sentences: int = 6
    max_sentences: int = 26
    resolve_contradictions: bool = True


class ReportDecoder:
    """Prototype bank + calibrated thresholds; the only object inference needs."""

    def __init__(
        self,
        bank: PrototypeBank,
        thresholds: Sequence[float] | np.ndarray,
        *,
        settings: DecoderSettings | None = None,
    ) -> None:
        thresholds = np.asarray(thresholds, dtype=np.float32)
        if thresholds.shape != (len(bank),):
            raise ValueError(f"thresholds must have shape ({len(bank)},), got {thresholds.shape}")
        self.bank = bank
        self.thresholds = thresholds
        self.settings = settings or DecoderSettings()
        self._order = {index: position for position, index in enumerate(bank.render_order)}
        self._contradictions = (
            contradiction_pairs(bank) if self.settings.resolve_contradictions else set()
        )

    # -- selection ---------------------------------------------------------

    def select(self, probabilities: Sequence[float] | np.ndarray) -> list[int]:
        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if values.shape != self.thresholds.shape:
            raise ValueError(
                f"probabilities must have shape {self.thresholds.shape}, got {values.shape}"
            )
        margin = values - self.thresholds
        selected = [int(index) for index in np.flatnonzero(margin >= 0)]

        if len(selected) < self.settings.min_sentences:
            # A report of zero characters scores zero; never emit fewer than the
            # floor even when the encoder is uncertain about everything.
            ranked = np.argsort(-margin)
            selected = sorted({*selected, *(int(i) for i in ranked[: self.settings.min_sentences])})
        if len(selected) > self.settings.max_sentences:
            selected = sorted(selected, key=lambda index: -margin[index])[
                : self.settings.max_sentences
            ]

        if self._contradictions:
            selected = self._resolve(selected, values)
        return sorted(selected, key=lambda index: self._order.get(index, len(self._order)))

    def _resolve(self, selected: Sequence[int], values: np.ndarray) -> list[int]:
        keep = set(selected)
        for left, right in self._contradictions:
            if left in keep and right in keep:
                keep.discard(right if values[left] >= values[right] else left)
        return sorted(keep)

    # -- rendering ---------------------------------------------------------

    def render(self, indices: Sequence[int]) -> str:
        ordered = sorted(indices, key=lambda index: self._order.get(index, len(self._order)))
        return join_report([self.bank[index].text for index in ordered])

    def decode(self, probabilities: Sequence[float] | np.ndarray) -> str:
        return self.render(self.select(probabilities))

    def decode_batch(self, probabilities: np.ndarray) -> list[str]:
        matrix = np.asarray(probabilities, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2D probability matrix, got shape {matrix.shape}")
        return [self.decode(row) for row in matrix]

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "decoder_version": DECODER_VERSION,
                    "thresholds": [float(value) for value in self.thresholds],
                    "min_sentences": self.settings.min_sentences,
                    "max_sentences": self.settings.max_sentences,
                    "resolve_contradictions": self.settings.resolve_contradictions,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def load(cls, path: str | Path, bank: PrototypeBank) -> ReportDecoder:
        location = Path(path)
        if not location.is_file():
            raise FileNotFoundError(
                f"Decoder not found at {location}. Run `cbct-reasoner calibrate` first."
            )
        payload = json.loads(location.read_text(encoding="utf-8-sig"))
        if int(payload.get("decoder_version", 0)) != DECODER_VERSION:
            raise ValueError(
                f"Decoder version {payload.get('decoder_version')} != "
                f"{DECODER_VERSION}; recalibrate."
            )
        return cls(
            bank,
            payload["thresholds"],
            settings=DecoderSettings(
                min_sentences=int(payload["min_sentences"]),
                max_sentences=int(payload["max_sentences"]),
                resolve_contradictions=bool(payload.get("resolve_contradictions", True)),
            ),
        )
