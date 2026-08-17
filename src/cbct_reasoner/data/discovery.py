"""Dataset discovery for the released ToothFairy4 layout and common variants.

The organizer's pipeline documents ``{CASE_ID}/cbct/volume.nii.gz`` with sibling
``reports_en/`` and ``reports_it/`` directories. Real downloads arrive wrapped in
an extra folder, occasionally flattened, and sometimes already converted to
``.mha``. ``discover_cases`` accepts each of those shapes but still fails loudly
on a case that has a volume without text or text without a volume, so training
data can never be silently dropped.

``inspect_layout`` is the read-only counterpart: it reports what was found and
what is wrong without raising, which is the right tool immediately after
extracting an archive.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cbct_reasoner.schemas import CaseRecord
from cbct_reasoner.text import normalize_text

VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")

#: Per-case volume locations, most specific first.
VOLUME_CANDIDATES = (
    "cbct/volume.nii.gz",
    "cbct/volume.nii",
    "cbct/volume.mha",
    "images/cbct/volume.nii.gz",
    "volume.nii.gz",
    "volume.mha",
)
REPORT_DIR_CANDIDATES = (
    "reports_en",
    "reports_english",
    "report_en",
    "reports/en",
    "reports/english",
)
#: Directories that wrap the real case folders in a downloaded archive.
WRAPPER_NAMES = frozenset(
    {"train", "training", "trainingset", "toothfairy4", "cases", "dataset", "data", "raw"}
)


def infer_center(case_id: str) -> str:
    """Infer the ToothFairy subset/center prefix (P, F, S, A) from a case identifier."""
    match = re.match(r"([A-Za-z]+)", case_id.strip())
    return match.group(1).upper() if match else "UNKNOWN"


def normalize_report(text: str) -> str:
    """Normalize whitespace while preserving clinical punctuation and qualifiers."""
    return normalize_text(text)


@dataclass(frozen=True, slots=True)
class LayoutReport:
    """Non-raising summary of a candidate dataset directory."""

    root: Path
    complete_cases: tuple[str, ...]
    volume_only: tuple[str, ...]
    reports_only: tuple[str, ...]
    empty_reports: tuple[str, ...]
    centers: dict[str, int]
    report_counts: dict[int, int]
    detected_layout: str

    @property
    def ok(self) -> bool:
        return bool(self.complete_cases) and not (self.volume_only or self.reports_only)

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "detected_layout": self.detected_layout,
            "complete_cases": len(self.complete_cases),
            "volume_without_reports": list(self.volume_only[:20]),
            "reports_without_volume": list(self.reports_only[:20]),
            "empty_report_dirs": list(self.empty_reports[:20]),
            "cases_by_center": self.centers,
            "reports_per_case_histogram": {
                str(k): v for k, v in sorted(self.report_counts.items())
            },
            "ok": self.ok,
        }

    def render(self) -> str:
        lines = [
            f"dataset root      : {self.root}",
            f"detected layout   : {self.detected_layout}",
            f"complete cases    : {len(self.complete_cases)}",
            f"cases by center   : {self.centers or '-'}",
            f"reports per case  : {dict(sorted(self.report_counts.items())) or '-'}",
        ]
        if self.volume_only:
            lines.append(
                f"MISSING REPORTS   : {len(self.volume_only)} e.g. {list(self.volume_only[:5])}"
            )
        if self.reports_only:
            lines.append(
                f"MISSING VOLUMES   : {len(self.reports_only)} e.g. {list(self.reports_only[:5])}"
            )
        if self.empty_reports:
            lines.append(
                f"EMPTY REPORT DIRS : {len(self.empty_reports)} e.g. {list(self.empty_reports[:5])}"
            )
        if not self.complete_cases:
            lines.append(
                "No cases found. Expected {CASE}/cbct/volume.nii.gz with {CASE}/reports_en/*.txt"
            )
        return "\n".join(lines)


def resolve_root(dataset_root: str | Path) -> Path:
    """Descend through single-child wrapper directories created by archive tools."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    for _ in range(4):
        children = [
            child for child in root.iterdir() if child.is_dir() and not child.name.startswith(".")
        ]
        looks_like_cases = any(_case_volume(child) is not None for child in children)
        if looks_like_cases or len(children) != 1:
            break
        if children[0].name.lower() in WRAPPER_NAMES or len(children) == 1:
            root = children[0]
    return root


def inspect_layout(dataset_root: str | Path) -> LayoutReport:
    """Describe a dataset directory without raising, for post-download triage."""
    root = resolve_root(dataset_root)
    complete: list[str] = []
    volume_only: list[str] = []
    reports_only: list[str] = []
    empty_reports: list[str] = []
    centers: Counter[str] = Counter()
    report_counts: Counter[int] = Counter()

    flat = _flat_layout(root)
    layout = "flat volumes/ + reports_en/" if flat else "per-case directories"

    for case_id, volume, reports in _iterate_candidates(root, flat):
        if volume is None and not reports:
            continue
        if volume is None:
            reports_only.append(case_id)
            continue
        if reports is None:
            volume_only.append(case_id)
            continue
        if not reports:
            empty_reports.append(case_id)
            continue
        complete.append(case_id)
        centers[infer_center(case_id)] += 1
        report_counts[len(reports)] += 1

    return LayoutReport(
        root=root,
        complete_cases=tuple(sorted(complete)),
        volume_only=tuple(sorted(volume_only)),
        reports_only=tuple(sorted(reports_only)),
        empty_reports=tuple(sorted(empty_reports)),
        centers=dict(sorted(centers.items())),
        report_counts=dict(report_counts),
        detected_layout=layout,
    )


def discover_cases(dataset_root: str | Path, *, require_reports: bool = True) -> list[CaseRecord]:
    """Discover complete ToothFairy4 cases.

    Set ``require_reports=False`` for an inference-only directory (the hidden
    test set ships volumes with no text).
    """
    root = resolve_root(dataset_root)
    flat = _flat_layout(root)

    records: list[CaseRecord] = []
    errors: list[str] = []
    for case_id, volume, reports in _iterate_candidates(root, flat):
        if volume is None and reports is None:
            continue
        if volume is None:
            if require_reports:
                errors.append(f"{case_id}: missing CBCT volume")
            continue
        if not require_reports:
            records.append(
                CaseRecord(
                    case_id=case_id, volume_path=volume, reports=("",), center=infer_center(case_id)
                )
            )
            continue
        if reports is None:
            errors.append(f"{case_id}: missing English report directory")
            continue
        if not reports:
            errors.append(f"{case_id}: no non-empty English .txt reports")
            continue
        records.append(
            CaseRecord(
                case_id=case_id,
                volume_path=volume,
                reports=reports,
                center=infer_center(case_id),
            )
        )

    if errors:
        preview = "; ".join(errors[:8])
        suffix = f"; and {len(errors) - 8} more" if len(errors) > 8 else ""
        raise ValueError(
            f"Invalid dataset cases: {preview}{suffix}. "
            f"Run `cbct-reasoner inspect --data {dataset_root}` for a full layout report."
        )
    if not records:
        raise ValueError(
            f"No ToothFairy4 cases found under {root}. "
            "Expected {CASE}/cbct/volume.nii.gz alongside {CASE}/reports_en/*.txt"
        )
    return sorted(records, key=lambda item: item.case_id.casefold())


def write_manifest(records: Iterable[CaseRecord], destination: str | Path) -> Path:
    """Write a portable JSONL manifest without copying protected clinical text."""
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


def read_manifest(source: str | Path) -> list[dict[str, object]]:
    path = Path(source)
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# Layout probing helpers
# --------------------------------------------------------------------------


def _flat_layout(root: Path) -> tuple[Path, Path] | None:
    """Detect a ``volumes/`` + ``reports_en/`` sibling layout."""
    for volume_dir_name in ("volumes", "cbct", "images"):
        volume_dir = root / volume_dir_name
        if not volume_dir.is_dir():
            continue
        if not any(_has_volume_suffix(path) for path in volume_dir.iterdir() if path.is_file()):
            continue
        for report_dir_name in ("reports_en", "reports_english", "reports"):
            report_dir = root / report_dir_name
            if report_dir.is_dir():
                return volume_dir, report_dir
    return None


def _iterate_candidates(
    root: Path, flat: tuple[Path, Path] | None
) -> Iterable[tuple[str, Path | None, tuple[str, ...] | None]]:
    if flat is not None:
        volume_dir, report_dir = flat
        volumes = {
            _strip_suffix(path.name): path
            for path in sorted(volume_dir.iterdir())
            if path.is_file() and _has_volume_suffix(path)
        }
        reports: dict[str, list[Path]] = {}
        for path in sorted(report_dir.rglob("*.txt")):
            reports.setdefault(_strip_suffix(path.name).split("_report")[0], []).append(path)
        for case_id in sorted(set(volumes) | set(reports)):
            texts = _read_reports(reports.get(case_id, [])) if case_id in reports else None
            yield case_id, volumes.get(case_id), texts
        return

    for case_dir in _case_directories(root):
        volume = _case_volume(case_dir)
        report_dir = _first_existing_dir(case_dir, REPORT_DIR_CANDIDATES)
        if report_dir is None:
            loose = sorted(case_dir.glob("*.txt"))
            texts = _read_reports(loose) if loose else None
        else:
            texts = _read_reports(sorted(report_dir.glob("*.txt")))
        yield case_dir.name, volume, texts


def _case_directories(root: Path) -> list[Path]:
    direct = sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda item: item.name,
    )
    if any(_case_volume(path) is not None for path in direct):
        return direct
    nested: list[Path] = []
    for wrapper in direct:
        if wrapper.name.lower() in WRAPPER_NAMES:
            nested.extend(path for path in wrapper.iterdir() if path.is_dir())
    return direct + sorted(nested, key=lambda item: item.name)


def _case_volume(case_dir: Path) -> Path | None:
    explicit = next(
        (case_dir / name for name in VOLUME_CANDIDATES if (case_dir / name).is_file()), None
    )
    if explicit is not None:
        return explicit
    for sub in ("cbct", "images/cbct", "image", "."):
        directory = case_dir / sub if sub != "." else case_dir
        if not directory.is_dir():
            continue
        matches = sorted(
            path for path in directory.iterdir() if path.is_file() and _has_volume_suffix(path)
        )
        if matches:
            return matches[0]
    return None


def _read_reports(paths: list[Path]) -> tuple[str, ...]:
    return tuple(
        normalized
        for path in paths
        if (normalized := normalize_report(path.read_text(encoding="utf-8-sig", errors="replace")))
    )


def _has_volume_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in VOLUME_SUFFIXES)


def _strip_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in (*VOLUME_SUFFIXES, ".txt"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _first_existing_dir(root: Path, candidates: tuple[str, ...]) -> Path | None:
    return next((root / candidate for candidate in candidates if (root / candidate).is_dir()), None)
