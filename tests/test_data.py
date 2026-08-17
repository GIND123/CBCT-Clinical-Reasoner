from pathlib import Path

import pytest

from cbct_reasoner.data import discover_cases, infer_center, write_manifest


def create_case(root: Path, case_id: str, report: str = "No focal lesion.") -> None:
    case = root / case_id
    (case / "cbct").mkdir(parents=True)
    (case / "cbct" / "volume.nii.gz").touch()
    (case / "reports_en").mkdir()
    (case / "reports_en" / "report_1.txt").write_text(report, encoding="utf-8")


def test_discovers_and_sorts_cases(tmp_path: Path) -> None:
    create_case(tmp_path, "P002")
    create_case(tmp_path, "A001", "  Implant site is adequate.\n")

    records = discover_cases(tmp_path)

    assert [record.case_id for record in records] == ["A001", "P002"]
    assert records[0].center == "A"
    assert records[0].reports == ("Implant site is adequate.",)


def test_rejects_incomplete_case(tmp_path: Path) -> None:
    case = tmp_path / "S001" / "cbct"
    case.mkdir(parents=True)
    (case / "volume.nii.gz").touch()

    with pytest.raises(ValueError, match="missing English report"):
        discover_cases(tmp_path)


def test_manifest_does_not_contain_report_text(tmp_path: Path) -> None:
    create_case(tmp_path, "F001", "Sensitive report text.")
    manifest = write_manifest(discover_cases(tmp_path), tmp_path / "out" / "manifest.jsonl")

    content = manifest.read_text(encoding="utf-8")
    assert "Sensitive report text" not in content
    assert '"report_count": 1' in content


@pytest.mark.parametrize(("case_id", "expected"), [("A003", "A"), ("p12", "P"), ("123", "UNKNOWN")])
def test_infer_center(case_id: str, expected: str) -> None:
    assert infer_center(case_id) == expected
