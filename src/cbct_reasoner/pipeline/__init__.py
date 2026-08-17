"""Pipeline stages and the deployable inference bundle."""

from cbct_reasoner.pipeline.bundle import InferenceBundle
from cbct_reasoner.pipeline.stages import (
    ablation,
    calibrate_decoder,
    evaluate,
    figures,
    load_oof,
    package,
    prepare,
    prototypes,
    splits,
    train,
)

__all__ = [
    "InferenceBundle",
    "ablation",
    "calibrate_decoder",
    "evaluate",
    "figures",
    "load_oof",
    "package",
    "prepare",
    "prototypes",
    "splits",
    "train",
]
