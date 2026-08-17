from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CaseRecord:
    """One CBCT case and its one-or-more English reference reports."""

    case_id: str
    volume_path: Path
    reports: tuple[str, ...]
    center: str

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty")
        if not self.reports:
            raise ValueError(f"case {self.case_id!r} has no reports")
        if any(not report.strip() for report in self.reports):
            raise ValueError(f"case {self.case_id!r} has an empty report")


@dataclass(frozen=True, slots=True)
class Volume:
    """In-memory CBCT array plus physical spacing in array-axis (z, y, x) order."""

    array: object
    spacing_zyx: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Prediction:
    """A generated report and retrieval provenance for auditing."""

    report: str
    source_case_id: str
    distance: float

    def to_output(self) -> dict[str, str]:
        """Return the exact Grand Challenge output payload."""
        return {"report": self.report}
