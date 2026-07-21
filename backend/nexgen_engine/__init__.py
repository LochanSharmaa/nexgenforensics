"""Commercial NexGen facial recognition engine."""

from .config import EngineConfig
from .inference.pipeline import FacialRecognitionPipeline

__all__ = ["EngineConfig", "FacialRecognitionPipeline"]
