import numpy as np
import pytest

from cbct_reasoner.features import extract_features
from cbct_reasoner.schemas import Volume


def test_features_are_fixed_size_and_finite() -> None:
    array = np.arange(12 * 10 * 8, dtype=np.float32).reshape(12, 10, 8)
    features = extract_features(Volume(array=array, spacing_zyx=(0.3, 0.3, 0.4)))

    assert features.shape == (85,)
    assert np.all(np.isfinite(features))


def test_features_reject_invalid_volume() -> None:
    with pytest.raises(ValueError, match="3D volume"):
        extract_features(Volume(array=np.zeros((4, 4)), spacing_zyx=(1, 1, 1)))


def test_features_ignore_isolated_non_finite_values() -> None:
    array = np.ones((4, 4, 4), dtype=np.float32)
    array[0, 0, 0] = np.nan
    array[0, 0, 1] = np.inf

    features = extract_features(Volume(array=array, spacing_zyx=(1, 1, 1)))

    assert np.all(np.isfinite(features))
