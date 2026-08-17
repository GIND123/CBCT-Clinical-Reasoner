from __future__ import annotations

import os
from pathlib import Path

from cbct_reasoner.model import RetrievalReportModel
from cbct_reasoner.reporting import write_challenge_output

DEFAULT_INPUT_DIR = Path("/input/images/cbct")
DEFAULT_OUTPUT = Path("/output/diagnostic-imaging-report.json")
DEFAULT_MODEL = Path("/opt/ml/model/retrieval_model.npz")


def find_single_cbct(input_dir: Path) -> Path:
    candidates = sorted(
        path
        for pattern in ("*.mha", "*.mhd", "*.nii", "*.nii.gz")
        for path in input_dir.glob(pattern)
        if path.is_file()
    )
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one CBCT in {input_dir}, found {len(candidates)}")
    return candidates[0]


def run(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    model_path: Path = DEFAULT_MODEL,
) -> Path:
    cbct = find_single_cbct(input_dir)
    prediction = RetrievalReportModel.load(model_path).predict_path(cbct)
    print(
        f"Retrieved report from training case {prediction.source_case_id}; "
        f"distance={prediction.distance:.6f}"
    )
    return write_challenge_output(prediction.report, output_path)


def main() -> int:
    run(
        input_dir=Path(os.getenv("CBCT_INPUT_DIR", str(DEFAULT_INPUT_DIR))),
        output_path=Path(os.getenv("REPORT_OUTPUT_PATH", str(DEFAULT_OUTPUT))),
        model_path=Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL))),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
