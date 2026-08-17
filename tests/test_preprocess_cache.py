"""Cache-geometry guards.

Regression: an interrupted preprocessing run left entries written on a different
grid. They loaded fine individually and only failed later inside the DataLoader
collate as an opaque "stack expects each tensor to be equal size" error, after a
GPU job had already started. A cache entry now records the grid it was written
on, and both the writer and the reader check it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cbct_reasoner.config import PreprocessConfig
from cbct_reasoner.data.preprocess import VolumeMeta, is_cached, read_cache, write_cache

CONFIG = PreprocessConfig(spacing_mm=(0.5, 0.5, 0.5), shape_zyx=(8, 12, 12))
OTHER = PreprocessConfig(spacing_mm=(0.6, 0.6, 0.6), shape_zyx=(10, 16, 16))


def make_meta(config: PreprocessConfig) -> VolumeMeta:
    return VolumeMeta(
        case_id="P001",
        original_shape_zyx=(200, 300, 300),
        original_spacing_zyx=(0.3, 0.3, 0.3),
        physical_size_mm=(60.0, 90.0, 90.0),
        intensity_low=-1000.0,
        intensity_high=2500.0,
        foreground_fraction=0.05,
        cached_shape_zyx=tuple(config.shape_zyx),
        cached_spacing_mm=tuple(config.spacing_mm),
    )


def write(tmp_path: Path, config: PreprocessConfig) -> None:
    write_cache(tmp_path, "P001", np.zeros(config.shape_zyx, dtype=np.float16), make_meta(config))


def test_cache_round_trips_its_grid(tmp_path: Path) -> None:
    write(tmp_path, CONFIG)
    array, meta = read_cache(tmp_path, "P001")

    assert array.shape == CONFIG.shape_zyx
    assert meta.cached_shape_zyx == CONFIG.shape_zyx
    assert meta.cached_spacing_mm == CONFIG.spacing_mm


def test_is_cached_accepts_a_matching_grid(tmp_path: Path) -> None:
    write(tmp_path, CONFIG)
    assert is_cached(tmp_path, "P001")
    assert is_cached(tmp_path, "P001", CONFIG)


def test_is_cached_rejects_a_different_shape(tmp_path: Path) -> None:
    write(tmp_path, CONFIG)
    assert not is_cached(tmp_path, "P001", OTHER)


def test_is_cached_rejects_a_different_spacing(tmp_path: Path) -> None:
    write(tmp_path, CONFIG)
    respaced = PreprocessConfig(spacing_mm=(0.8, 0.8, 0.8), shape_zyx=CONFIG.shape_zyx)
    assert not is_cached(tmp_path, "P001", respaced)


def test_is_cached_rejects_a_stale_version(tmp_path: Path) -> None:
    write(tmp_path, CONFIG)
    meta_path = tmp_path / "P001.json"
    meta_path.write_text(meta_path.read_text().replace('"cache_version": 3', '"cache_version": 1'))

    assert not is_cached(tmp_path, "P001", CONFIG)


def test_missing_case_is_not_cached(tmp_path: Path) -> None:
    assert not is_cached(tmp_path, "P001")
    assert not is_cached(tmp_path, "P001", CONFIG)


def test_dataset_rejects_a_mismatched_shape(tmp_path: Path) -> None:
    """Fail at load with the case id, not inside collate with a bare size error."""
    torch = pytest.importorskip("torch")
    assert torch is not None
    from cbct_reasoner.data.dataset import CBCTDataset

    write(tmp_path, CONFIG)
    dataset = CBCTDataset(["P001"], tmp_path, expected_shape=OTHER.shape_zyx)

    with pytest.raises(ValueError, match="was cached at"):
        dataset[0]

    assert CBCTDataset(["P001"], tmp_path, expected_shape=CONFIG.shape_zyx)[0]["volume"].shape == (
        1,
        *CONFIG.shape_zyx,
    )
