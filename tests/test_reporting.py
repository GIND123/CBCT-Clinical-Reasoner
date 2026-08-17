import json
from pathlib import Path

import pytest

from cbct_reasoner.grand_challenge import find_single_cbct
from cbct_reasoner.reporting import read_challenge_output, write_challenge_output


def test_output_contract_round_trip(tmp_path: Path) -> None:
    output = write_challenge_output("  No focal lesion.\n", tmp_path / "report.json")

    assert json.loads(output.read_text(encoding="utf-8")) == {"report": "No focal lesion."}
    assert read_challenge_output(output) == "No focal lesion."


def test_output_rejects_extra_field(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"report": "text", "debug": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        read_challenge_output(output)


def test_find_single_cbct(tmp_path: Path) -> None:
    volume = tmp_path / "sample.mha"
    volume.touch()
    assert find_single_cbct(tmp_path) == volume


def test_find_single_cbct_rejects_ambiguous_input(tmp_path: Path) -> None:
    (tmp_path / "one.mha").touch()
    (tmp_path / "two.mha").touch()
    with pytest.raises(RuntimeError, match="exactly one"):
        find_single_cbct(tmp_path)
