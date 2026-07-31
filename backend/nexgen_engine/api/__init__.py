"""Engine-facing service layer.

This package had no __init__.py, so `nexgen_engine.api.service` was not
importable as a package module on all import paths.
"""

from .schemas import EngineMatch, EngineSearchResponse

__all__ = ["EngineMatch", "EngineSearchResponse"]
