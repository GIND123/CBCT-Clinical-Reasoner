from __future__ import annotations

from pathlib import Path

import numpy as np

from cbct_reasoner.schemas import Volume

FEATURE_VERSION = "global-statistics-v1"
QUANTILES = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])


def load_volume(path: str | Path) -> Volume:
    """Read a NIfTI or MHA volume with SimpleITK and return z-y-x ordered data."""
    try:
        import SimpleITK as sitk
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("SimpleITK is required to read CBCT volumes") from error

    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"CBCT volume does not exist: {location}")
    image = sitk.ReadImage(str(location))
    array = sitk.GetArrayFromImage(image)
    spacing_xyz = image.GetSpacing()
    return Volume(
        array=np.asarray(array, dtype=np.float32),
        spacing_zyx=tuple(float(value) for value in reversed(spacing_xyz)),
    )


def extract_features(volume: Volume) -> np.ndarray:
    """Extract deterministic, bounded-memory global CBCT retrieval features.

    This intentionally modest feature encoder is an executable baseline, not a claim
    of clinical understanding. A production entry should replace it with anatomy-aware
    segmentation/detection and an ontology-grounded report decoder.
    """
    array = np.asarray(volume.array, dtype=np.float32)
    if array.ndim != 3 or any(size < 2 for size in array.shape):
        raise ValueError(f"Expected a non-degenerate 3D volume, got shape {array.shape}")
    if len(volume.spacing_zyx) != 3 or any(value <= 0 for value in volume.spacing_zyx):
        raise ValueError(f"Expected three positive spacing values, got {volume.spacing_zyx}")

    sampled = _bounded_sample(array)
    finite = sampled[np.isfinite(sampled)]
    if finite.size < 16:
        raise ValueError("CBCT contains fewer than 16 finite sampled voxels")

    quantiles = np.quantile(finite, QUANTILES).astype(np.float32)
    low, high = float(quantiles[0]), float(quantiles[-1])
    scale = max(high - low, 1e-6)
    cleaned = np.nan_to_num(sampled, nan=low, posinf=high, neginf=low)
    normalized = np.clip((cleaned - low) / scale, 0, 1)

    histogram, _ = np.histogram(normalized, bins=16, range=(0.0, 1.0), density=False)
    histogram = histogram.astype(np.float32) / max(float(histogram.sum()), 1.0)
    normalized_quantiles = ((quantiles - low) / scale).astype(np.float32)
    mean = float(np.mean(normalized))
    std = max(float(np.std(normalized)), 1e-6)
    skew = float(np.mean(((normalized - mean) / std) ** 3))

    shape = np.asarray(array.shape, dtype=np.float32)
    spacing = np.asarray(volume.spacing_zyx, dtype=np.float32)
    physical_size = shape * spacing
    geometry = np.concatenate([np.log1p(shape), np.log1p(spacing), np.log1p(physical_size)])
    moments = np.asarray([mean, std, np.clip(skew, -10, 10)], dtype=np.float32)
    projections = np.concatenate(
        [_pool_2d(normalized.mean(axis=axis), bins=4) for axis in range(3)]
    ).astype(np.float32)

    features = np.concatenate(
        [geometry, normalized_quantiles, moments, histogram, projections]
    ).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature extraction produced non-finite values")
    return features


def _bounded_sample(array: np.ndarray, maximum_axis: int = 96) -> np.ndarray:
    indices = [
        np.linspace(0, size - 1, min(size, maximum_axis), dtype=np.int64)
        for size in array.shape
    ]
    return array[np.ix_(*indices)]


def _pool_2d(array: np.ndarray, bins: int) -> np.ndarray:
    rows = np.array_split(array, bins, axis=0)
    pooled: list[float] = []
    for row in rows:
        pooled.extend(float(np.mean(cell)) for cell in np.array_split(row, bins, axis=1))
    return np.asarray(pooled, dtype=np.float32)
