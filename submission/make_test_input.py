"""Create a Grand Challenge-shaped test input directory.

Converts a local NIfTI case to the ``.mha`` the platform mounts, or synthesizes a
volume when no dataset is available yet:

    python submission/make_test_input.py                       # synthetic
    python submission/make_test_input.py --case data/raw/P001  # from the release
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "submission" / "test" / "input"


def synthesize(case_id: str) -> sitk.Image:
    rng = np.random.default_rng(2026)
    shape = (160, 240, 240)
    volume = rng.normal(0.25, 0.06, size=shape).astype(np.float32)
    z = np.linspace(-1, 1, shape[0])[:, None, None]
    y = np.linspace(-1, 1, shape[1])[None, :, None]
    x = np.linspace(-1, 1, shape[2])[None, None, :]
    volume += 0.7 * np.exp(-((y - 0.15) ** 2 + x**2) * 5.0) * np.exp(-(z**2) * 2.5)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((0.3, 0.3, 0.3))
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, help="a case directory containing cbct/volume.nii.gz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-id", default="P001")
    args = parser.parse_args()

    case_id = args.case.name if args.case else args.case_id
    if args.case:
        source = next(iter(sorted((args.case / "cbct").glob("*.nii*"))), None)
        if source is None:
            raise SystemExit(f"No NIfTI volume under {args.case / 'cbct'}")
        image = sitk.ReadImage(str(source))
    else:
        image = synthesize(case_id)

    images_dir = args.output / "images" / "cbct"
    images_dir.mkdir(parents=True, exist_ok=True)
    destination = images_dir / f"{case_id}.mha"
    sitk.WriteImage(image, str(destination), useCompression=True)

    (args.output / "inputs.json").write_text(
        json.dumps(
            [
                {
                    "socket": {
                        "slug": "cbct-image",
                        "relative_path": "images/cbct",
                        "is_image_kind": True,
                        "is_panimg_kind": True,
                        "is_dicom_image_kind": False,
                        "is_json_kind": False,
                        "is_file_kind": False,
                    },
                    "file": None,
                    "image": {"name": f"{case_id}.mha"},
                    "value": None,
                }
            ],
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {destination} and {args.output / 'inputs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
