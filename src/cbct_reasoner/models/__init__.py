"""Predictors.

Imports are lazy on purpose. ``shallow`` needs only numpy, and the slim
submission image ships without torch to serve a 1 MB model from a 625 MB image
instead of a 12 GB one. Eagerly importing the torch-backed modules here made
``from cbct_reasoner.models.shallow import ...`` raise ``ModuleNotFoundError`` in
that image, which the bundle caught as a generic failure and silently answered
with the prior report - a worse model shipping while looking healthy.
"""

from typing import TYPE_CHECKING, Any

# Re-exported for type checkers only; `__all__` is built from _EXPORTS below, so
# these look unused to a linter.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from cbct_reasoner.models.losses import (  # noqa: F401
        AsymmetricLoss,
        ModelEma,
        average_precision,
    )
    from cbct_reasoner.models.network import (  # noqa: F401
        FindingNet,
        GatedAttentionPool,
        ResNet3dBackbone,
        Slice2dBackbone,
        build_network,
    )
    from cbct_reasoner.models.shallow import (  # noqa: F401
        ShallowConfig,
        ShallowModel,
        build_features,
        fit_full,
        fit_out_of_fold,
        volume_descriptor,
    )
    from cbct_reasoner.models.trainer import (  # noqa: F401
        FoldResult,
        load_checkpoint,
        predict_probabilities,
        resolve_device,
        save_checkpoint,
        train_fold,
    )

#: attribute -> submodule. Only the submodule actually asked for is imported.
_EXPORTS: dict[str, str] = {
    "AsymmetricLoss": "losses",
    "ModelEma": "losses",
    "average_precision": "losses",
    "FindingNet": "network",
    "GatedAttentionPool": "network",
    "ResNet3dBackbone": "network",
    "Slice2dBackbone": "network",
    "build_network": "network",
    "ShallowConfig": "shallow",
    "ShallowModel": "shallow",
    "build_features": "shallow",
    "fit_full": "shallow",
    "fit_out_of_fold": "shallow",
    "volume_descriptor": "shallow",
    "FoldResult": "trainer",
    "load_checkpoint": "trainer",
    "predict_probabilities": "trainer",
    "resolve_device": "trainer",
    "save_checkpoint": "trainer",
    "train_fold": "trainer",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module_name}"), name)


def __dir__() -> list[str]:
    return __all__
