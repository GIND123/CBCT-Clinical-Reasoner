"""Regression tests for the container entrypoint.

A submission scored BLEU 0.0161 because the container could not load its bundle
and silently emitted a generic paragraph for all 50 test cases. The failure was
invisible locally, cost a whole submission slot, and is the reason the entrypoint
now has four independent layers. These tests exercise each layer by breaking the
one above it, so a future edit cannot quietly remove the safety net.

They also pin the two properties that make the geometry path safe to run at all:
it must never decode pixel data, and it must never import torch.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

SUBMISSION = Path(__file__).resolve().parents[1] / "submission"


def load_entrypoint(monkeypatch, tmp_path: Path):
    """Import submission/inference.py with its I/O paths redirected."""
    monkeypatch.syspath_prepend(str(SUBMISSION))
    spec = importlib.util.spec_from_file_location("gc_inference", SUBMISSION / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INPUT_PATH = tmp_path / "input"
    module.OUTPUT_PATH = tmp_path / "output"
    module.MODEL_PATHS = (tmp_path / "absent",)
    return module


def write_volume(root: Path, shape_zyx=(200, 300, 300), spacing=(0.3, 0.3, 0.3)) -> Path:
    location = root / "input" / "images" / "cbct"
    location.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.zeros(shape_zyx, dtype=np.int16))
    image.SetSpacing(tuple(float(s) for s in spacing[::-1]))
    destination = location / "case.mha"
    sitk.WriteImage(image, str(destination), useCompression=True)
    return destination


def read_output(module) -> str:
    payload = json.loads((module.OUTPUT_PATH / "diagnostic-imaging-report.json").read_text())
    return payload["report"]


def fake_adaptive() -> dict:
    """A two-gate model whose gates key on volume height, so geometry decides."""
    return {
        "version": 1,
        "texts": ["Core statement.", "Tall-volume statement.", "Short-volume statement."],
        "core": [0],
        "conditional": [1, 2],
        # Feature 0 is log1p(size_z); +/- weight flips the two gates apart.
        "coefficients": [[4.0] + [0.0] * 8, [-4.0] + [0.0] * 8],
        "intercepts": [0.0, 0.0],
        "thresholds": [0.5, 0.5],
        "mean": [math.log1p(250.0)] + [0.0] * 8,
        "scale": [1.0] * 9,
        "order": [0, 1, 2],
    }


def test_report_is_written_without_any_input(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch, tmp_path)
    assert module.run() == 0
    report = read_output(module)
    assert report == " ".join(module.CONSTANT_REPORT.split())
    assert len(report) > 200


def test_geometry_changes_the_report(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch, tmp_path)
    module.ADAPTIVE = fake_adaptive()

    write_volume(tmp_path, shape_zyx=(400, 300, 300))
    assert module.run() == 0
    tall = read_output(module)

    write_volume(tmp_path, shape_zyx=(100, 300, 300))
    assert module.run() == 0
    short = read_output(module)

    assert tall != short
    assert "Core statement." in tall and "Core statement." in short
    assert "Tall-volume statement." in tall
    assert "Short-volume statement." in short


def test_unreadable_volume_falls_back_to_the_constant_report(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch, tmp_path)
    module.ADAPTIVE = fake_adaptive()
    location = tmp_path / "input" / "images" / "cbct"
    location.mkdir(parents=True)
    (location / "case.mha").write_bytes(b"not an image at all")

    assert module.run() == 0
    assert read_output(module) == " ".join(module.CONSTANT_REPORT.split())


def test_missing_adaptive_model_falls_back_to_the_constant_report(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch, tmp_path)
    module.ADAPTIVE = None
    write_volume(tmp_path)

    assert module.run() == 0
    assert read_output(module) == " ".join(module.CONSTANT_REPORT.split())


def test_output_is_still_written_when_the_report_path_is_hostile(monkeypatch, tmp_path):
    """The final layer: write_report raising must not produce an empty /output."""
    module = load_entrypoint(monkeypatch, tmp_path)
    original = module.write_report
    calls = {"n": 0}

    def flaky(report: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated read-only filesystem")
        original(report)

    module.write_report = flaky
    assert module.run() == 0
    assert read_output(module) == module.FALLBACK_REPORT


def test_geometry_path_never_imports_torch(monkeypatch, tmp_path):
    """torch is not in the slim image; importing it would abort every case."""
    module = load_entrypoint(monkeypatch, tmp_path)
    module.ADAPTIVE = fake_adaptive()
    write_volume(tmp_path)

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("the geometry path must not import torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert module.run() == 0
    assert "Core statement." in read_output(module)


def test_header_read_does_not_decode_pixels(monkeypatch, tmp_path):
    """GetArrayFromImage on a full read would be the expensive, fragile path."""
    volume = write_volume(tmp_path, shape_zyx=(400, 400, 400))
    module = load_entrypoint(monkeypatch, tmp_path)

    monkeypatch.setattr(
        sitk,
        "ReadImage",
        lambda *a, **k: pytest.fail("header path must not call ReadImage"),
    )
    features = module.header_features(volume)
    assert features is not None and len(features) == 9
    assert features[0] == pytest.approx(math.log1p(400.0))
    assert features[3] == pytest.approx(0.3)


def test_embedded_model_matches_the_fitted_artifact():
    """report_model.py is generated; a stale checkout should not ship silently."""
    generated = SUBMISSION / "report_model.py"
    if not generated.is_file():
        pytest.skip("report_model.py not built yet")
    artifact = Path(__file__).resolve().parents[1] / "artifacts/adaptive_report.json"
    if not artifact.is_file():
        pytest.skip("no fitted adaptive report to compare against")

    spec = importlib.util.spec_from_file_location("gc_report_model", generated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fitted = json.loads(artifact.read_text(encoding="utf-8"))

    assert module.ADAPTIVE is not None, "a fitted report exists but was not embedded"
    assert len(module.ADAPTIVE["conditional"]) == len(fitted["conditional"])
    assert module.ADAPTIVE["thresholds"] == fitted["thresholds"]
    used = set(fitted["core"]) | set(fitted["conditional"])
    assert len(module.ADAPTIVE["texts"]) == len(used)
