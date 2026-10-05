"""AI Meeting Action-Item Extractor."""
from .pipeline import PipelineResult, run_pipeline
from .schema import ActionItem

__all__ = ["run_pipeline", "PipelineResult", "ActionItem"]
__version__ = "1.0.0"
