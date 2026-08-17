from pathlib import Path

import numpy as np

from cbct_reasoner.model import RetrievalReportModel, select_consensus_report
from cbct_reasoner.schemas import CaseRecord, Volume


def volume(value: float) -> Volume:
    base = np.linspace(0, 1, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    return Volume(array=base + value, spacing_zyx=(0.3 + value, 0.3, 0.3))


def test_model_round_trip_and_prediction(tmp_path: Path) -> None:
    records = [
        CaseRecord("A001", Path("first"), ("First report.",), "A"),
        CaseRecord("P001", Path("second"), ("Second report.",), "P"),
    ]
    volumes = {Path("first"): volume(0.0), Path("second"): volume(0.8)}
    model = RetrievalReportModel.fit(records, volume_loader=volumes.__getitem__)
    artifact = model.save(tmp_path / "model.npz")

    prediction = RetrievalReportModel.load(artifact).predict_volume(volume(0.75))

    assert prediction.report == "Second report."
    assert prediction.source_case_id == "P001"
    assert prediction.distance >= 0


def test_consensus_report_selects_shared_vocabulary() -> None:
    reports = (
        "Impacted tooth 48 close to the mandibular canal.",
        "Tooth 48 is impacted and close to mandibular canal.",
        "Unrelated short note.",
    )

    assert select_consensus_report(reports) != "Unrelated short note."
