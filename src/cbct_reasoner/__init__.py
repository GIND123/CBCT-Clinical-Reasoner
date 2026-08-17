"""ToothFairy4 CBCT clinical report generation (ODIN 2026 Task 1).

Pipeline: normalize the volume, predict which reportable findings it shows, then
render only the findings whose emission raises the expected challenge score.
"""

from cbct_reasoner.config import ExperimentConfig, Paths, default_paths
from cbct_reasoner.model import RetrievalReportModel
from cbct_reasoner.schemas import CaseRecord, Prediction

__all__ = [
    "CaseRecord",
    "ExperimentConfig",
    "Paths",
    "Prediction",
    "RetrievalReportModel",
    "default_paths",
]
__version__ = "0.2.0"
