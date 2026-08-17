"""Synthetic ToothFairy4-shaped data for end-to-end rehearsal.

The real release is access-controlled and cannot live in this repository, but
every stage of the pipeline still has to be provably wired before the data
arrives. This module fabricates cases whose reports are generated from a latent
finding vector that is *correlated with the image*, so training on it produces a
model that measurably beats the prior - proving the supervision path is
connected rather than just that the code runs.

Nothing here is clinical. The sentences are plausible-sounding filler used only
to exercise tokenizers, clustering, and metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: (sentence template, base prevalence, image-signal weight). Prevalence spans
#: near-universal normals down to rare findings, matching real report statistics.
TEMPLATES: tuple[tuple[str, float, float], ...] = (
    ("Cone-beam CT examination of the maxillofacial region was performed.", 0.98, 0.0),
    ("The mandibular canal is identifiable bilaterally along its entire course.", 0.88, 0.1),
    ("The maxillary sinuses are symmetrically pneumatized.", 0.72, 0.2),
    ("No periapical radiolucency is observed.", 0.62, -0.4),
    ("Periapical radiolucency is present at the apex of tooth 46.", 0.18, 0.9),
    ("Mucosal thickening is noted in the left maxillary sinus.", 0.24, 0.8),
    ("Mucosal thickening is noted in the right maxillary sinus.", 0.22, 0.75),
    ("The third molar 48 is impacted and in close proximity to the mandibular canal.", 0.20, 0.95),
    ("The alveolar ridge shows moderate vertical resorption in the posterior mandible.", 0.30, 0.7),
    ("Residual bone height above the sinus floor measures approximately 6 mm.", 0.26, 0.6),
    ("Endodontic treatment is present in the posterior maxilla.", 0.28, 0.5),
    ("A dental implant is present in the posterior mandible.", 0.16, 0.85),
    ("The temporomandibular joints show no gross degenerative change.", 0.55, -0.3),
    ("Condylar flattening is observed on the right side.", 0.14, 0.8),
    ("The nasopalatine canal is of normal calibre.", 0.48, 0.15),
    ("Cortical bone thickness at the implant site appears adequate.", 0.34, 0.45),
    ("No fracture line is identified.", 0.66, -0.25),
    ("Bone density in the posterior maxilla corresponds to type III bone.", 0.22, 0.6),
    ("Multiple coronal restorations are present.", 0.40, 0.4),
    ("The edentulous posterior mandible shows reduced bone width.", 0.20, 0.7),
)


def _latent_to_probabilities(latent: np.ndarray) -> np.ndarray:
    base = np.asarray([item[1] for item in TEMPLATES], dtype=np.float64)
    weight = np.asarray([item[2] for item in TEMPLATES], dtype=np.float64)
    logits = np.log(base / (1 - base)) + weight * latent * 3.0
    return 1.0 / (1.0 + np.exp(-logits))


def _render(active: np.ndarray) -> str:
    sentences = [TEMPLATES[index][0] for index in np.flatnonzero(active)]
    if not sentences:
        sentences = [TEMPLATES[0][0], "No significant abnormality is identified."]
    return " ".join(sentences)


def generate_dataset(
    destination: str | Path,
    *,
    num_cases: int = 24,
    shape: tuple[int, int, int] = (24, 32, 32),
    seed: int = 7,
    centers: tuple[str, ...] = ("P", "F", "S", "A"),
) -> Path:
    """Write a ToothFairy4-shaped tree of NIfTI volumes and English reports."""
    import SimpleITK as sitk

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for index in range(num_cases):
        center = centers[index % len(centers)]
        case_id = f"{center}{index + 1:03d}"
        case_dir = root / case_id
        (case_dir / "cbct").mkdir(parents=True, exist_ok=True)
        (case_dir / "reports_en").mkdir(parents=True, exist_ok=True)

        # One latent factor drives both the image texture and the findings, so a
        # model that reads the image can beat a model that only knows the prior.
        latent = float(rng.normal())
        volume = rng.normal(loc=0.25, scale=0.08, size=shape).astype(np.float32)
        z = np.linspace(-1, 1, shape[0])[:, None, None]
        y = np.linspace(-1, 1, shape[1])[None, :, None]
        x = np.linspace(-1, 1, shape[2])[None, None, :]
        arch = np.exp(-((y - 0.2 * latent) ** 2 + x**2) * 4.0) * np.exp(-(z**2) * 2.0)
        volume += (0.6 + 0.3 * latent) * arch.astype(np.float32)

        image = sitk.GetImageFromArray(volume)
        image.SetSpacing((0.3, 0.3, 0.3))
        sitk.WriteImage(image, str(case_dir / "cbct" / "volume.nii.gz"))

        probabilities = _latent_to_probabilities(np.full(len(TEMPLATES), latent))
        num_reports = int(rng.choice([1, 2, 2, 3], p=[0.4, 0.3, 0.2, 0.1]))
        for report_index in range(num_reports):
            active = rng.random(len(TEMPLATES)) < probabilities
            (case_dir / "reports_en" / f"report_{report_index + 1}.txt").write_text(
                _render(active), encoding="utf-8"
            )
    return root
