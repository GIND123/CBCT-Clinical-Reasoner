from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from cbct_reasoner.schemas import CaseRecord

VOLUME_CANDIDATES = (
    "cbct/volume.nii.gz",
    "images/cbct/volume.nii.gz",
    "volume.nii.gz",
)
REPORT_DIR_CANDIDATES = ("reports_en", "report_en", "reports/english")


def infer_center(case_id: str) -> str:
    """Infer the public ToothFairy subset/center prefix from a case identifier."""
    match = re.match(r"([A-Za-z]+)", case_id.strip())
    return match.group(1).upper() if match else "UNKNOWN"


def discover_cases(dataset_root: str | Path) -> list[CaseRecord]:
    """Discover cases in the released ToothFairy4 directory layout.

    The loader is deliberately tolerant of an optional wrapper directory, but strict
    about missing volumes or English reports so silent training-data loss is impossible.
    """
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    records: list[CaseRecord] = []
    errors: list[str] = []
    for case_dir in _candidate_case_directories(root):
        volume = _first_existing(case_dir, VOLUME_CANDIDATES)
        report_dir = _first_existing_dir(case_dir, REPORT_DIR_CANDIDATES)
        if volume is None and report_dir is None:
            continue
        if volume is None:
            errors.append(f"{case_dir.name}: missing CBCT volume")
            continue
        if report_dir is None:
            errors.append(f"{case_dir.name}: missing English report directory")
            continue

        report_paths = sorted(report_dir.glob("*.txt"))
        reports = tuple(
            normalized
            for path in report_paths
            if (normalized := normalize_report(path.read_text(encoding="utf-8-sig")))
        )
        if not reports:
            errors.append(f"{case_dir.name}: no non-empty English .txt reports")
            continue
        records.append(
            CaseRecord(
                case_id=case_dir.name,
                volume_path=volume,
                reports=reports,
                center=infer_center(case_dir.name),
            )
        )

    if errors:
        preview = "; ".join(errors[:8])
        suffix = f"; and {len(errors) - 8} more" if len(errors) > 8 else ""
        raise ValueError(f"Invalid dataset cases: {preview}{suffix}")
    if not records:
        raise ValueError(f"No ToothFairy4 cases found under {root}")
    return sorted(records, key=lambda item: item.case_id.casefold())


def write_manifest(records: Iterable[CaseRecord], destination: str | Path) -> Path:
    """Write a portable JSONL manifest without copying protected clinical data."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.parent.resolve()
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            try:
                volume = str(record.volume_path.resolve().relative_to(root))
            except ValueError:
                volume = str(record.volume_path.resolve())
            stream.write(
                json.dumps(
                    {
                        "case_id": record.case_id,
                        "center": record.center,
                        "volume": volume,
                        "report_count": len(record.reports),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output


def normalize_report(text: str) -> str:
    """Normalize whitespace while preserving clinical punctuation and qualifiers."""
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def _candidate_case_directories(root: Path) -> list[Path]:
    direct = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
    nested: list[Path] = []
    for wrapper in direct:
        if wrapper.name.lower() in {"train", "training", "toothfairy4", "cases"}:
            nested.extend(path for path in wrapper.iterdir() if path.is_dir())
    return direct + sorted(nested, key=lambda p: p.name)


def _first_existing(root: Path, candidates: tuple[str, ...]) -> Path | None:
    return next(
        (root / candidate for candidate in candidates if (root / candidate).is_file()),
        None,
    )


def _first_existing_dir(root: Path, candidates: tuple[str, ...]) -> Path | None:
    return next((root / candidate for candidate in candidates if (root / candidate).is_dir()), None)
