"""Full-pipeline rehearsal on synthetic data.

Proves the wiring end to end - discovery, caching, label construction, training,
out-of-fold assembly, calibration, packaging, and container-shaped inference -
before the real release is available. Marked ``slow`` but still runs in seconds
on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cbct_reasoner import pipeline
from cbct_reasoner.config import ExperimentConfig, Paths
from cbct_reasoner.data.synthetic import generate_dataset
from cbct_reasoner.pipeline.bundle import InferenceBundle

pytest.importorskip("SimpleITK")
pytest.importorskip("sklearn")

CONFIG = ExperimentConfig.from_dict(
    {
        "name": "test",
        "seed": 1,
        "folds": 3,
        "preprocess": {"spacing_mm": [1.0, 1.0, 1.0], "shape_zyx": [16, 24, 24]},
        "prototypes": {"max_prototypes": 32, "min_support": 2},
        "encoder": {
            "backbone": "resnet3d",
            "width": 8,
            "epochs": 2,
            "batch_size": 2,
            "accumulate": 1,
            "num_workers": 0,
            "amp": False,
        },
        "decode": {"min_sentences": 3, "max_sentences": 12, "calibration_rounds": 1},
    }
)


def FIRST_CASE_VOLUME(paths: Paths) -> Path:
    """Any real case. Case IDs cycle P/F/S/A, so a hardcoded 'A001' does not exist."""
    case = next(path for path in sorted(paths.data.iterdir()) if path.is_dir())
    return case / "cbct" / "volume.nii.gz"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Paths:
    root = tmp_path_factory.mktemp("toothfairy4")
    generate_dataset(root / "raw", num_cases=24, shape=(16, 24, 24), seed=5)
    return Paths(
        root=root, data=root / "raw", work=root / "work", artifacts=root / "artifacts"
    ).ensure()


@pytest.mark.slow
def test_full_pipeline(workspace: Paths) -> None:
    prepared = pipeline.prepare(workspace, CONFIG)
    assert prepared["cases"] == 24
    assert prepared["cached"] == 24
    assert prepared["failures"] == []

    built = pipeline.prototypes(workspace, CONFIG)
    assert built["num_prototypes"] > 3
    # The bank should explain nearly every phrase the corpus contains; low
    # coverage means findings exist that the decoder can never emit.
    assert built["phrase_coverage"] > 0.8

    plan = pipeline.splits(workspace, CONFIG)
    assert len(plan["folds"]) == 3
    assert sum(fold["validation"] for fold in plan["folds"]) == 24

    trained = pipeline.train(workspace, CONFIG, device="cpu")
    assert trained["coverage"] == 1.0

    case_ids, probabilities, covered = pipeline.load_oof(workspace)
    assert probabilities.shape == (24, built["num_prototypes"])
    assert covered.all()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    calibrated = pipeline.calibrate_decoder(workspace, CONFIG)
    assert 0.0 <= calibrated["objective"]["final"] <= 1.0
    assert calibrated["objective"]["final"] >= calibrated["baseline"]["final"] - 1e-9

    scored = pipeline.evaluate(workspace, CONFIG)
    assert scored["num_cases"] == 24
    assert scored["surrogate"]["bleu_4"] > 0.0

    packaged = pipeline.package(workspace, CONFIG)
    bundle_dir = Path(packaged["bundle"])
    for name in (
        "prototypes.json",
        "decoder.json",
        "config.json",
        "fallback_report.txt",
        "bundle.json",
    ):
        assert (bundle_dir / name).is_file()
    assert len(list((bundle_dir / "checkpoints").glob("fold*.pt"))) == 3


@pytest.mark.slow
def test_bundle_reads_the_image_rather_than_falling_back(workspace: Paths) -> None:
    """The report must come from the checkpoints, not from the prior.

    Asserting only that some text was produced would pass even if every volume
    silently failed to load and the fallback was returned instead.
    """
    bundle = InferenceBundle.load(workspace.bundle)
    volume = FIRST_CASE_VOLUME(workspace)
    assert volume.is_file()
    assert len(bundle.checkpoints) == 3

    probabilities = bundle.probabilities(volume)
    report = bundle.predict(volume)

    # Image-conditioned output differs from the raw corpus prior.
    assert probabilities.shape == bundle.bank.prevalence.shape
    assert not np.allclose(probabilities, bundle.bank.prevalence)
    assert report.strip()
    assert report.endswith(".")
    assert len(report.split()) > 10


@pytest.mark.slow
def test_two_different_volumes_can_produce_different_reports(workspace: Paths) -> None:
    """A decoder that emits one constant report for every scan is not using the image."""
    bundle = InferenceBundle.load(workspace.bundle)
    cases = sorted(path for path in workspace.data.iterdir() if path.is_dir())

    vectors = [bundle.probabilities(case / "cbct" / "volume.nii.gz") for case in cases[:6]]
    spread = np.std(np.stack(vectors), axis=0).max()

    assert spread > 0.0


@pytest.mark.slow
def test_bundle_falls_back_instead_of_raising(workspace: Paths, tmp_path: Path) -> None:
    """A missing result scores zero, so an unreadable volume must still emit text."""
    bundle = InferenceBundle.load(workspace.bundle)
    broken = tmp_path / "not-a-volume.nii.gz"
    broken.write_bytes(b"this is not a NIfTI file")

    report = bundle.predict(broken)

    assert report == bundle.fallback_report
    assert report.strip()


@pytest.mark.slow
def test_prior_only_bundle_needs_no_checkpoints(workspace: Paths, tmp_path: Path) -> None:
    from cbct_reasoner.decode.decoder import ReportDecoder
    from cbct_reasoner.prototypes import PrototypeBank

    bank = PrototypeBank.load(workspace.prototypes)
    decoder = ReportDecoder.load(workspace.decoder, bank)
    InferenceBundle.write(tmp_path / "bundle", bank=bank, decoder=decoder, config=CONFIG)

    bundle = InferenceBundle.load(tmp_path / "bundle")
    report = bundle.predict(FIRST_CASE_VOLUME(workspace))

    assert bundle.checkpoints == ()
    assert report.strip()


@pytest.mark.slow
def test_container_entrypoint_writes_the_exact_contract(workspace: Paths, tmp_path: Path) -> None:
    import SimpleITK as sitk

    from cbct_reasoner.grand_challenge import run

    volume = FIRST_CASE_VOLUME(workspace)
    source = sitk.ReadImage(str(volume))
    input_dir = tmp_path / "input" / "images" / "cbct"
    input_dir.mkdir(parents=True)
    sitk.WriteImage(source, str(input_dir / f"{volume.parent.parent.name}.mha"))

    output = tmp_path / "output" / "diagnostic-imaging-report.json"
    run(input_dir=input_dir, output_path=output, model_path=workspace.bundle)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload) == {"report"}
    assert isinstance(payload["report"], str)
    assert payload["report"].strip()
