"""World models for Experiment 3 (structured vs flat prediction)."""

from src.models.structured_model import StructuredWorldModel, GaussianMLP
from src.models.flat_model import FlatWorldModel
from src.models.ensemble import ModelEnsemble

__all__ = ["StructuredWorldModel", "GaussianMLP", "FlatWorldModel", "ModelEnsemble"]
