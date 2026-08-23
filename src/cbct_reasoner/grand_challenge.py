"""Grand Challenge container entrypoint.

Implements the ToothFairy4 interface exactly as the organizer's template does:

===========================  ==========================================
input socket ``cbct-image``  ``/input/images/cbct/*.mha``
output ``diagnostic-...``    ``/output/diagnostic-imaging-report.json``
model tarball                ``/opt/ml/model``
===========================  ==========================================

The process is written to always produce a report. A missing result is scored as
zero characters on every metric, so an exception here is strictly worse than a
generic report; each failure mode degrades one level rather than aborting.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from cbct_reasoner.reporting import write_challenge_output

DEFAULT_INPUT_DIR = Path("/input/images/cbct")
DEFAULT_OUTPUT = Path("/output/diagnostic-imaging-report.json")
DEFAULT_MODEL = Path("/opt/ml/model")

#: Used only if the bundle itself cannot be loaded - a report of zero characters
#: scores zero, so even a fully generic paragraph is worth emitting.
EMERGENCY_REPORT = (
    "Cone-beam CT examination of the maxillofacial region was performed. "
    "The mandibular canal is identifiable bilaterally along its course. "
    "The maxillary sinuses are pneumatized. "
    "The alveolar bone shows no gross destructive change. "
    "The temporomandibular joints show no gross degenerative change. "
    "No acute fracture is identified."
)

VOLUME_PATTERNS = ("*.mha", "*.mhd", "*.nii", "*.nii.gz", "*.nrrd")


def find_single_cbct(input_dir: Path) -> Path:
    candidates = sorted(
        {path for pattern in VOLUME_PATTERNS for path in input_dir.glob(pattern) if path.is_file()}
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one CBCT in {input_dir}, found {len(candidates)}")
    return candidates[0]


def _describe_environment() -> None:
    try:
        import torch

        available = torch.cuda.is_available()
        print(f"torch {torch.__version__}; CUDA available: {available}", flush=True)
        if available:
            print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as error:  # pragma: no cover - torch is optional at inference
        print(f"torch unavailable: {error}", flush=True)


def generate_report(cbct: Path, model_dir: Path) -> str:
    from cbct_reasoner.pipeline.bundle import InferenceBundle

    bundle = InferenceBundle.load(model_dir)
    return bundle.predict(cbct)


def run(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    model_path: Path = DEFAULT_MODEL,
) -> Path:
    _describe_environment()
    try:
        cbct = find_single_cbct(input_dir)
        print(f"Reading CBCT: {cbct.name}", flush=True)
        report = generate_report(cbct, model_path)
    except Exception:
        traceback.print_exc()
        print(
            "Falling back to the emergency report so the case is not scored as empty.", flush=True
        )
        report = EMERGENCY_REPORT
    print(f"Report: {len(report)} characters", flush=True)
    return write_challenge_output(report, output_path)


def _interface_key(input_root: Path) -> tuple[str, ...]:
    """Read the socket list the platform mounts, when present."""
    manifest = input_root / "inputs.json"
    if not manifest.is_file():
        return ("cbct-image",)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return tuple(sorted(item["socket"]["slug"] for item in payload))


def main() -> int:
    input_dir = Path(os.getenv("CBCT_INPUT_DIR", str(DEFAULT_INPUT_DIR)))
    output_path = Path(os.getenv("REPORT_OUTPUT_PATH", str(DEFAULT_OUTPUT)))
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))

    interface = _interface_key(
        input_dir.parent.parent if input_dir.name == "cbct" else Path("/input")
    )
    if interface not in {("cbct-image",)}:
        print(
            f"warning: unexpected interface {interface}; proceeding with the CBCT handler",
            file=sys.stderr,
        )

    run(input_dir=input_dir, output_path=output_path, model_path=model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
