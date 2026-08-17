"""The text side of the dataset: reports, phrases, and reference selection.

The hidden test set supplies exactly one reference report per case, but training
cases carry up to three. Which one you train and calibrate against materially
changes the score, so the choice is made with the grading metric itself: the
retained reference is the *METEOR medoid*, the report a grader would score
highest on average against the alternatives.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cbct_reasoner.metrics.official import meteor_score_tokenized, tokenize
from cbct_reasoner.schemas import CaseRecord
from cbct_reasoner.text import is_verifiable, split_phrases


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One case's text: every reference, plus the selected consensus reference."""

    case_id: str
    center: str
    reports: tuple[str, ...]
    reference: str
    phrases: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "center": self.center,
            "reports": list(self.reports),
            "reference": self.reference,
            "phrases": list(self.phrases),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CorpusEntry:
        return cls(
            case_id=str(payload["case_id"]),
            center=str(payload["center"]),
            reports=tuple(str(item) for item in payload["reports"]),  # type: ignore[union-attr]
            reference=str(payload["reference"]),
            phrases=tuple(str(item) for item in payload["phrases"]),  # type: ignore[union-attr]
        )

    @property
    def all_phrases(self) -> tuple[str, ...]:
        """Verifiable phrases across every reference for this case.

        RadFact recall is measured against one reference, but a finding recorded
        by any annotator is genuinely present in the scan, so the *union* is the
        right supervision target for the imaging model.
        """
        seen: dict[str, None] = {}
        for report in self.reports:
            for phrase in split_phrases(report):
                if is_verifiable(phrase):
                    seen.setdefault(phrase, None)
        return tuple(seen)


def select_reference(reports: Sequence[str]) -> str:
    """Pick the report with the highest mean METEOR against the other references.

    With one report the choice is trivial; with two, the longer one wins ties
    because METEOR is recall-weighted and the grader's single hidden reference is
    more likely to be covered by the more complete text.
    """
    if not reports:
        raise ValueError("reports cannot be empty")
    if len(reports) == 1:
        return reports[0]

    tokenized = [tokenize(report) for report in reports]
    scores: list[float] = []
    for index, candidate in enumerate(tokenized):
        others = [other for position, other in enumerate(tokenized) if position != index]
        scores.append(
            sum(meteor_score_tokenized(candidate, other) for other in others) / len(others)
        )
    best = max(range(len(reports)), key=lambda index: (scores[index], len(tokenized[index])))
    return reports[best]


def build_corpus(records: Iterable[CaseRecord]) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    for record in records:
        reference = select_reference(record.reports)
        phrases = tuple(phrase for phrase in split_phrases(reference) if is_verifiable(phrase))
        entries.append(
            CorpusEntry(
                case_id=record.case_id,
                center=record.center,
                reports=record.reports,
                reference=reference,
                phrases=phrases,
            )
        )
    return entries


def save_corpus(entries: Iterable[CorpusEntry], destination: str | Path) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for entry in entries:
            stream.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return output


def load_corpus(source: str | Path) -> list[CorpusEntry]:
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(
            f"Report corpus not found at {path}. Run `cbct-reasoner prepare` first."
        )
    entries: list[CorpusEntry] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                entries.append(CorpusEntry.from_dict(json.loads(line)))
    if not entries:
        raise ValueError(f"Report corpus at {path} is empty")
    return entries


def corpus_index(entries: Iterable[CorpusEntry]) -> dict[str, CorpusEntry]:
    return {entry.case_id: entry for entry in entries}
