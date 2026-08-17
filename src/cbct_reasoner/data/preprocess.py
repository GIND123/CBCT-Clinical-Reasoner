"""CBCT normalization and voxel caching.

Every volume is brought onto one physical grid before it reaches the encoder:
canonical orientation, fixed millimetre spacing, then a crop centred on the
jaws. Spacing is preserved rather than resizing the whole field of view, because
CBCT fields of view range from a single quadrant to the full skull and the
reports quantify real distances ("2.1 mm of residual bone"). A network trained on
scale-normalized volumes cannot learn those.

Caches are written as ``float16`` ``.npy`` next to a small JSON sidecar. They are
derived clinical data and stay under ``work/`` which is git-ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cbct_reasoner.config import PreprocessConfig

CACHE_VERSION = 2


@dataclass(frozen=True, slots=True)
class VolumeMeta:
    """Geometry of the original acquisition, used as auxiliary model input."""

    case_id: str
    original_shape_zyx: tuple[int, int, int]
    original_spacing_zyx: tuple[float, float, float]
    physical_size_mm: tuple[float, float, float]
    intensity_low: float
    intensity_high: float
    foreground_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_version": CACHE_VERSION,
            "case_id": self.case_id,
            "original_shape_zyx": list(self.original_shape_zyx),
            "original_spacing_zyx": list(self.original_spacing_zyx),
            "physical_size_mm": list(self.physical_size_mm),
            "intensity_low": self.intensity_low,
            "intensity_high": self.intensity_high,
            "foreground_fraction": self.foreground_fraction,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VolumeMeta:
        return cls(
            case_id=str(payload["case_id"]),
            original_shape_zyx=tuple(int(v) for v in payload["original_shape_zyx"]),  # type: ignore[arg-type]
            original_spacing_zyx=tuple(float(v) for v in payload["original_spacing_zyx"]),  # type: ignore[arg-type]
            physical_size_mm=tuple(float(v) for v in payload["physical_size_mm"]),  # type: ignore[arg-type]
            intensity_low=float(payload["intensity_low"]),
            intensity_high=float(payload["intensity_high"]),
            foreground_fraction=float(payload["foreground_fraction"]),
        )

    def to_vector(self) -> np.ndarray:
        """Compact geometry descriptor appended to the pooled image embedding."""
        shape = np.asarray(self.original_shape_zyx, dtype=np.float32)
        spacing = np.asarray(self.original_spacing_zyx, dtype=np.float32)
        physical = np.asarray(self.physical_size_mm, dtype=np.float32)
        return np.concatenate(
            [
                np.log1p(shape) / 8.0,
                spacing,
                np.log1p(physical) / 8.0,
                np.asarray([self.foreground_fraction], dtype=np.float32),
            ]
        ).astype(np.float32)


META_DIM = 10


def preprocess_volume(
    path: str | Path, config: PreprocessConfig, *, case_id: str | None = None
) -> tuple[np.ndarray, VolumeMeta]:
    """Load, orient, resample, crop, and intensity-normalize one CBCT."""
    import SimpleITK as sitk

    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"CBCT volume does not exist: {location}")

    image = sitk.ReadImage(str(location))
    if image.GetDimension() != 3:
        raise ValueError(f"{location} is not a 3D volume (dimension {image.GetDimension()})")

    original_spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    original_size_xyz = tuple(int(v) for v in image.GetSize())

    try:
        image = sitk.DICOMOrient(image, config.orientation)
    except RuntimeError:
        # Some CBCT exports carry a degenerate direction matrix; identity is the
        # documented fallback and keeps the case in the training set.
        image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    image = _resample_to_spacing(image, config.spacing_mm, sitk)
    array = sitk.GetArrayFromImage(image).astype(np.float32)  # z, y, x

    finite = array[np.isfinite(array)]
    if finite.size < 64:
        raise ValueError(f"{location} contains too few finite voxels")
    low, high = (float(v) for v in np.percentile(finite, config.clip_percentiles))
    if high - low < 1e-3:
        high = low + 1.0
    array = np.clip(np.nan_to_num(array, nan=low, posinf=high, neginf=low), low, high)
    array = (array - low) / (high - low)

    foreground = array > 0.55
    foreground_fraction = float(foreground.mean())
    array = _center_crop_or_pad(array, config.shape_zyx, _dentition_mask(array, foreground))

    meta = VolumeMeta(
        case_id=case_id or location.stem,
        original_shape_zyx=(original_size_xyz[2], original_size_xyz[1], original_size_xyz[0]),
        original_spacing_zyx=(
            original_spacing_xyz[2],
            original_spacing_xyz[1],
            original_spacing_xyz[0],
        ),
        physical_size_mm=(
            original_size_xyz[2] * original_spacing_xyz[2],
            original_size_xyz[1] * original_spacing_xyz[1],
            original_size_xyz[0] * original_spacing_xyz[0],
        ),
        intensity_low=low,
        intensity_high=high,
        foreground_fraction=foreground_fraction,
    )
    return array.astype(np.dtype(config.dtype)), meta


def _resample_to_spacing(image, spacing_mm: tuple[float, float, float], sitk) -> Any:
    """Resample onto isotropic millimetre spacing (spacing_mm is given z, y, x)."""
    target_xyz = (float(spacing_mm[2]), float(spacing_mm[1]), float(spacing_mm[0]))
    current_xyz = tuple(float(v) for v in image.GetSpacing())
    current_size = tuple(int(v) for v in image.GetSize())
    if all(abs(a - b) < 1e-4 for a, b in zip(current_xyz, target_xyz, strict=True)):
        return image

    new_size = [
        max(1, int(round(size * current / target)))
        for size, current, target in zip(current_size, current_xyz, target_xyz, strict=True)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_xyz)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))
    return resampler.Execute(image)


def _dentition_mask(array: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Isolate enamel and restorations, the densest structures in a jaw CBCT.

    Centring on all bone puts the window at the skull's centre of mass on a
    large field of view, which drifts away from the dental arch the reports
    describe. Enamel is a far tighter landmark, and every case in this release
    has some. Falls back to bone when a scan is fully edentulous.
    """
    if not foreground.any():
        return foreground
    threshold = float(np.quantile(array[foreground], 0.90))
    dense = array >= max(threshold, 0.6)
    return dense if dense.sum() >= 512 else foreground


def _center_crop_or_pad(
    array: np.ndarray, shape: tuple[int, int, int], foreground: np.ndarray
) -> np.ndarray:
    """Crop around the dentition centroid, padding with air where the FOV is small."""
    centre = _foreground_centroid(foreground, array.shape)
    output = np.zeros(shape, dtype=array.dtype)
    slices_src: list[slice] = []
    slices_dst: list[slice] = []
    for axis, target in enumerate(shape):
        size = array.shape[axis]
        start = int(round(centre[axis] - target / 2))
        start = max(0, min(start, max(0, size - target)))
        length = min(target, size - start)
        offset = (target - length) // 2
        slices_src.append(slice(start, start + length))
        slices_dst.append(slice(offset, offset + length))
    output[tuple(slices_dst)] = array[tuple(slices_src)]
    return output


def _foreground_centroid(
    foreground: np.ndarray, shape: tuple[int, ...]
) -> tuple[float, float, float]:
    if not foreground.any():
        return tuple(size / 2 for size in shape)  # type: ignore[return-value]
    coords = [
        float(np.average(np.arange(shape[axis]), weights=weights))
        if (weights := foreground.sum(axis=tuple(i for i in range(3) if i != axis))).sum() > 0
        else shape[axis] / 2
        for axis in range(3)
    ]
    return tuple(coords)  # type: ignore[return-value]


def cache_paths(cache_dir: str | Path, case_id: str) -> tuple[Path, Path]:
    directory = Path(cache_dir)
    return directory / f"{case_id}.npy", directory / f"{case_id}.json"


def write_cache(
    cache_dir: str | Path, case_id: str, array: np.ndarray, meta: VolumeMeta
) -> tuple[Path, Path]:
    array_path, meta_path = cache_paths(cache_dir, case_id)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_name(array_path.name + ".tmp")
    # np.save() silently appends ".npy" to any path that lacks it, so the array
    # is written through an open handle to keep the temporary name exact.
    with temporary.open("wb") as stream:
        np.save(stream, array)
    temporary.replace(array_path)
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")
    return array_path, meta_path


def read_cache(cache_dir: str | Path, case_id: str) -> tuple[np.ndarray, VolumeMeta]:
    array_path, meta_path = cache_paths(cache_dir, case_id)
    if not array_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"No preprocessed cache for case {case_id!r} in {cache_dir}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(payload.get("cache_version", 0)) != CACHE_VERSION:
        raise ValueError(
            f"Cache for {case_id!r} was written by version {payload.get('cache_version')}, "
            f"expected {CACHE_VERSION}. Re-run `cbct-reasoner prepare --force`."
        )
    return np.load(array_path, mmap_mode="r"), VolumeMeta.from_dict(payload)


def is_cached(cache_dir: str | Path, case_id: str) -> bool:
    array_path, meta_path = cache_paths(cache_dir, case_id)
    return array_path.is_file() and meta_path.is_file()
