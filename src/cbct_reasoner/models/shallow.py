"""A low-capacity alternative to the fine-tuned encoder.

Fine-tuning a 29M-parameter backbone on 622 cases with ~1000 targets memorised
the training folds and generalised nothing: out-of-fold AUC came out at 0.486
prevalence-weighted, i.e. indistinguishable from the corpus prior, on statements
where a *ten-feature logistic regression over acquisition geometry alone* reaches
0.61-0.88.

That measurement is the whole argument for this module. The signal that is
actually recoverable at this sample size lives in coarse, global properties -
which jaw is in the field of view, how large the volume is, the gross intensity
distribution - not in fine texture that needs millions of parameters and tens of
thousands of cases to learn. So this fits one heavily-regularised linear model
per statement over a compact global descriptor, with the same grouped folds, and
emits out-of-fold probabilities in exactly the format the decoder and calibrator
already consume.

Statements with too few positives to fit are left at their corpus prior, which is
the best available estimate for something seen twice.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cbct_reasoner.data.preprocess import VolumeMeta, cache_paths
from cbct_reasoner.data.splits import SplitPlan

#: Quantiles of the normalized intensity distribution.
_QUANTILES = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])


@dataclass(frozen=True, slots=True)
class ShallowConfig:
    """Regularisation is deliberately strong; the sample size is the constraint."""

    min_support: int = 12
    inverse_regularization: float = 0.05
    max_iterations: int = 2000
    projection_bins: int = 4
    profile_bins: int = 12
    histogram_bins: int = 16


def volume_descriptor(array: np.ndarray, config: ShallowConfig) -> np.ndarray:
    """Global image descriptor: intensity shape plus coarse spatial occupancy.

    The spatial profiles matter most. "Mandibular condyles are not included in
    the scan" is a statement about where anatomy sits in the volume, and a
    per-axis occupancy profile captures exactly that.
    """
    data = np.asarray(array, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size < 16:
        raise ValueError("cached volume has too few finite voxels")

    quantiles = np.quantile(finite, _QUANTILES).astype(np.float32)
    histogram, _ = np.histogram(data, bins=config.histogram_bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32) / max(float(histogram.sum()), 1.0)

    mean = float(data.mean())
    std = max(float(data.std()), 1e-6)
    skew = float(np.clip(((data - mean) / std**3).mean(), -10, 10))

    bone = data > 0.55
    parts: list[np.ndarray] = [quantiles, histogram, np.asarray([mean, std, skew], np.float32)]

    for axis in range(3):
        others = tuple(index for index in range(3) if index != axis)
        # Where along this axis does dense bone sit, and how much of it?
        profile = bone.mean(axis=others).astype(np.float32)
        parts.append(_pool(profile, config.profile_bins))
        parts.append(_pool2d(data.mean(axis=axis), config.projection_bins))

    descriptor = np.concatenate(parts).astype(np.float32)
    return np.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)


def _pool(values: np.ndarray, bins: int) -> np.ndarray:
    return np.asarray([float(chunk.mean()) for chunk in np.array_split(values, bins)], np.float32)


def _pool2d(plane: np.ndarray, bins: int) -> np.ndarray:
    pooled: list[float] = []
    for row in np.array_split(plane, bins, axis=0):
        pooled.extend(float(cell.mean()) for cell in np.array_split(row, bins, axis=1))
    return np.asarray(pooled, dtype=np.float32)


def build_features(
    cache_dir: str | Path, case_ids: Sequence[str], config: ShallowConfig | None = None
) -> np.ndarray:
    """Concatenate the acquisition-geometry vector with the image descriptor."""
    settings = config or ShallowConfig()
    rows: list[np.ndarray] = []
    for case_id in case_ids:
        array_path, meta_path = cache_paths(cache_dir, case_id)
        meta = VolumeMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        array = np.load(array_path, mmap_mode="r")
        rows.append(
            np.concatenate([meta.to_vector(), volume_descriptor(np.asarray(array), settings)])
        )
    return np.vstack(rows).astype(np.float32)


def fit_out_of_fold(
    features: np.ndarray,
    labels: np.ndarray,
    case_ids: Sequence[str],
    plan: SplitPlan,
    prior: np.ndarray,
    config: ShallowConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-statement logistic regression, honest out-of-fold predictions.

    Uses the *same* folds as the neural run so the two are directly comparable.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    settings = config or ShallowConfig()
    position = {case_id: row for row, case_id in enumerate(case_ids)}
    probabilities = np.tile(prior.astype(np.float32), (len(case_ids), 1))
    fitted = 0

    for column in range(labels.shape[1]):
        target = labels[:, column].astype(int)
        if target.sum() < settings.min_support or target.sum() > len(target) - settings.min_support:
            continue  # left at the prior
        fitted += 1
        for fold in plan:
            train_rows = [position[c] for c in fold.train if c in position]
            validation_rows = [position[c] for c in fold.validation if c in position]
            y = target[train_rows]
            if y.sum() < 2 or y.sum() == len(y):
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=settings.inverse_regularization,
                    max_iter=settings.max_iterations,
                    class_weight="balanced",
                ),
            )
            model.fit(features[train_rows], y)
            probabilities[validation_rows, column] = model.predict_proba(features[validation_rows])[
                :, 1
            ].astype(np.float32)

    return probabilities, {
        "fitted_statements": float(fitted),
        "total_statements": float(labels.shape[1]),
        "feature_dimension": float(features.shape[1]),
    }


@dataclass(frozen=True, slots=True)
class ShallowModel:
    """Deployable form: standardization stats plus one linear model per statement.

    Only the statements that met the support floor are stored; every other
    prototype falls back to its corpus prior, which is exactly what the
    out-of-fold fit did, so training and inference agree by construction.
    """

    columns: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    prior: np.ndarray

    def predict(self, descriptor: np.ndarray) -> np.ndarray:
        values = np.asarray(descriptor, dtype=np.float64).reshape(-1)
        if values.shape != self.mean.shape:
            raise ValueError(f"descriptor must have shape {self.mean.shape}, got {values.shape}")
        standardized = (values - self.mean) / self.scale
        logits = self.coefficients @ standardized + self.intercepts
        probabilities = self.prior.astype(np.float32).copy()
        probabilities[self.columns] = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        return probabilities

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            columns=self.columns.astype(np.int32),
            coefficients=self.coefficients.astype(np.float32),
            intercepts=self.intercepts.astype(np.float32),
            mean=self.mean.astype(np.float32),
            scale=self.scale.astype(np.float32),
            prior=self.prior.astype(np.float32),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> ShallowModel:
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(
                columns=archive["columns"].astype(np.int64),
                coefficients=archive["coefficients"].astype(np.float64),
                intercepts=archive["intercepts"].astype(np.float64),
                mean=archive["mean"].astype(np.float64),
                scale=archive["scale"].astype(np.float64),
                prior=archive["prior"].astype(np.float32),
            )


def fit_full(
    features: np.ndarray,
    labels: np.ndarray,
    prior: np.ndarray,
    config: ShallowConfig | None = None,
) -> ShallowModel:
    """Refit on every case for deployment, using the settings validated out-of-fold."""
    from sklearn.linear_model import LogisticRegression

    settings = config or ShallowConfig()
    mean = features.mean(axis=0).astype(np.float64)
    scale = np.where(features.std(axis=0) > 1e-8, features.std(axis=0), 1.0).astype(np.float64)
    standardized = (features.astype(np.float64) - mean) / scale

    columns: list[int] = []
    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    for column in range(labels.shape[1]):
        target = labels[:, column].astype(int)
        if target.sum() < settings.min_support or target.sum() > len(target) - settings.min_support:
            continue
        model = LogisticRegression(
            C=settings.inverse_regularization,
            max_iter=settings.max_iterations,
            class_weight="balanced",
        )
        model.fit(standardized, target)
        columns.append(column)
        coefficients.append(model.coef_[0])
        intercepts.append(float(model.intercept_[0]))

    return ShallowModel(
        columns=np.asarray(columns, dtype=np.int64),
        coefficients=np.vstack(coefficients) if coefficients else np.zeros((0, features.shape[1])),
        intercepts=np.asarray(intercepts, dtype=np.float64),
        mean=mean,
        scale=scale,
        prior=np.asarray(prior, dtype=np.float32),
    )
