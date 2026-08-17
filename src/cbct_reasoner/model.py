from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from cbct_reasoner.features import FEATURE_VERSION, extract_features, load_volume
from cbct_reasoner.schemas import CaseRecord, Prediction, Volume

FeatureLoader = Callable[[Path], Volume]


class RetrievalReportModel:
    """A transparent nearest-neighbor baseline over global CBCT features."""

    def __init__(
        self,
        *,
        features: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        case_ids: tuple[str, ...],
        reports: tuple[str, ...],
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        center = np.asarray(center, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        if features.ndim != 2 or not len(features):
            raise ValueError("features must be a non-empty 2D matrix")
        if center.shape != (features.shape[1],) or scale.shape != center.shape:
            raise ValueError("normalization vectors do not match feature dimension")
        if len(case_ids) != len(features) or len(reports) != len(features):
            raise ValueError("case_ids and reports must match the number of feature rows")
        if np.any(scale <= 0) or not np.all(np.isfinite(features)):
            raise ValueError("model arrays contain invalid values")
        self.features = features
        self.center = center
        self.scale = scale
        self.case_ids = case_ids
        self.reports = reports

    @classmethod
    def fit(
        cls,
        records: Iterable[CaseRecord],
        *,
        volume_loader: FeatureLoader = load_volume,
    ) -> RetrievalReportModel:
        rows: list[np.ndarray] = []
        case_ids: list[str] = []
        reports: list[str] = []
        for record in records:
            rows.append(extract_features(volume_loader(record.volume_path)))
            case_ids.append(record.case_id)
            reports.append(select_consensus_report(record.reports))
        if not rows:
            raise ValueError("Cannot fit a model without cases")

        matrix = np.vstack(rows).astype(np.float32)
        center = np.median(matrix, axis=0).astype(np.float32)
        q75, q25 = np.quantile(matrix, [0.75, 0.25], axis=0)
        scale = np.asarray(q75 - q25, dtype=np.float32)
        standard_deviation = np.std(matrix, axis=0).astype(np.float32)
        scale = np.where(scale > 1e-6, scale, standard_deviation)
        scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        standardized = (matrix - center) / scale
        return cls(
            features=standardized,
            center=center,
            scale=scale,
            case_ids=tuple(case_ids),
            reports=tuple(reports),
        )

    def predict_volume(self, volume: Volume) -> Prediction:
        query = (extract_features(volume) - self.center) / self.scale
        distances = np.mean((self.features - query) ** 2, axis=1)
        index = int(np.argmin(distances))
        return Prediction(
            report=self.reports[index],
            source_case_id=self.case_ids[index],
            distance=float(distances[index]),
        )

    def predict_path(self, path: str | Path) -> Prediction:
        return self.predict_volume(load_volume(path))

    def save(self, destination: str | Path) -> Path:
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "format_version": 1,
                "feature_version": FEATURE_VERSION,
                "model_type": "nearest-neighbor-report-retrieval",
            }
        )
        with output.open("wb") as stream:
            np.savez_compressed(
                stream,
                features=self.features,
                center=self.center,
                scale=self.scale,
                case_ids=np.asarray(self.case_ids, dtype=np.str_),
                reports=np.asarray(self.reports, dtype=np.str_),
                metadata=np.asarray(metadata, dtype=np.str_),
            )
        return output

    @classmethod
    def load(cls, source: str | Path) -> RetrievalReportModel:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"features", "center", "scale", "case_ids", "reports", "metadata"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Model artifact is missing: {sorted(missing)}")
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format_version") != 1:
                raise ValueError(f"Unsupported model format: {metadata.get('format_version')}")
            if metadata.get("feature_version") != FEATURE_VERSION:
                raise ValueError("Model feature version does not match this package")
            return cls(
                features=archive["features"],
                center=archive["center"],
                scale=archive["scale"],
                case_ids=tuple(str(value) for value in archive["case_ids"].tolist()),
                reports=tuple(str(value) for value in archive["reports"].tolist()),
            )


def select_consensus_report(reports: tuple[str, ...]) -> str:
    """Choose the reference with greatest average token-set agreement."""
    if not reports:
        raise ValueError("reports cannot be empty")
    if len(reports) == 1:
        return reports[0]
    token_sets = [set(re.findall(r"\w+", report.casefold())) for report in reports]
    agreement: list[float] = []
    for index, tokens in enumerate(token_sets):
        scores = []
        for other_index, other in enumerate(token_sets):
            if index == other_index:
                continue
            union = tokens | other
            scores.append(len(tokens & other) / len(union) if union else 1.0)
        agreement.append(sum(scores) / len(scores))
    best = max(range(len(reports)), key=lambda index: (agreement[index], len(reports[index])))
    return reports[best]
