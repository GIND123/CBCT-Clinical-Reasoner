"""The low-capacity predictor and its deployment path.

Chosen over the fine-tuned encoder on measured out-of-fold evidence: the encoder
reached AUC 0.486 prevalence-weighted (chance) where this reaches 0.593, and
0.669 on the statements with enough support to be learnable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from cbct_reasoner.data.splits import stratified_group_folds  # noqa: E402
from cbct_reasoner.models.shallow import (  # noqa: E402
    ShallowConfig,
    ShallowModel,
    fit_full,
    fit_out_of_fold,
    volume_descriptor,
)

CONFIG = ShallowConfig(min_support=5)


def synthetic(seed: int = 0, cases: int = 80, statements: int = 6):
    """One latent factor drives both the features and half the statements."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=cases)
    features = np.column_stack(
        [latent + rng.normal(0, 0.3, cases), rng.normal(size=cases), rng.normal(size=cases)]
    ).astype(np.float32)
    labels = np.zeros((cases, statements), dtype=np.float32)
    for column in range(statements):
        if column % 2 == 0:  # learnable from the latent factor
            labels[:, column] = (latent + rng.normal(0, 0.4, cases) > 0).astype(np.float32)
        else:  # pure noise at a fixed base rate
            labels[:, column] = (rng.random(cases) < 0.4).astype(np.float32)
    prior = labels.mean(axis=0).astype(np.float32)
    case_ids = [f"P{i:03d}" for i in range(cases)]
    return features, labels, prior, case_ids


def test_descriptor_is_finite_and_fixed_length() -> None:
    rng = np.random.default_rng(1)
    volume = rng.random((16, 24, 24)).astype(np.float32)
    first = volume_descriptor(volume, CONFIG)
    second = volume_descriptor(rng.random((16, 24, 24)).astype(np.float32), CONFIG)

    assert first.shape == second.shape
    assert np.isfinite(first).all()


def test_out_of_fold_learns_only_the_learnable_statements() -> None:
    features, labels, prior, case_ids = synthetic()
    plan = stratified_group_folds(case_ids, ["P"] * len(case_ids), n_folds=4, seed=1)

    probabilities, info = fit_out_of_fold(features, labels, case_ids, plan, prior, CONFIG)

    assert probabilities.shape == labels.shape
    assert info["fitted_statements"] > 0

    def auc(scores, target):
        order = np.argsort(scores)
        ranks = np.empty(len(scores))
        ranks[order] = np.arange(1, len(scores) + 1)
        positives = target.sum()
        return (ranks[target == 1].sum() - positives * (positives + 1) / 2) / (
            positives * (len(target) - positives)
        )

    signal = np.mean([auc(probabilities[:, c], labels[:, c]) for c in (0, 2, 4)])
    noise = np.mean([auc(probabilities[:, c], labels[:, c]) for c in (1, 3, 5)])
    assert signal > 0.75
    assert abs(noise - 0.5) < 0.2


def test_unsupported_statements_stay_at_the_prior() -> None:
    features, labels, prior, case_ids = synthetic()
    labels[:, 1] = 0.0
    labels[:3, 1] = 1.0  # only 3 positives, below min_support
    plan = stratified_group_folds(case_ids, ["P"] * len(case_ids), n_folds=4, seed=1)

    probabilities, _ = fit_out_of_fold(features, labels, case_ids, plan, prior, CONFIG)

    assert np.allclose(probabilities[:, 1], prior[1])


def test_full_fit_round_trips_and_matches_prior_outside_fitted_columns(tmp_path: Path) -> None:
    features, labels, prior, _ = synthetic()
    model = fit_full(features, labels, prior, CONFIG)
    restored = ShallowModel.load(model.save(tmp_path / "shallow.npz"))

    predicted = model.predict(features[0])
    assert np.allclose(predicted, restored.predict(features[0]))
    assert predicted.shape == prior.shape
    assert ((predicted >= 0) & (predicted <= 1)).all()

    untouched = [c for c in range(labels.shape[1]) if c not in set(model.columns.tolist())]
    for column in untouched:
        assert predicted[column] == pytest.approx(prior[column])


def test_predict_rejects_a_mismatched_descriptor() -> None:
    features, labels, prior, _ = synthetic()
    model = fit_full(features, labels, prior, CONFIG)

    with pytest.raises(ValueError, match="descriptor must have shape"):
        model.predict(np.zeros(features.shape[1] + 5))
