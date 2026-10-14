"""Phase 4: Independent Speech Flaw Detection, Temporal Localization & Evidence."""

from flawbench.detection.engine import DetectionEngine
from flawbench.detection.inputs import build_detection_input, build_detection_input_from_features
from flawbench.detection.pipeline import DetectionPipeline
from flawbench.detection.schema import DetectionInput, DetectionResult, FlawPrediction
from flawbench.detection.store import DetectionStore
from flawbench.detection.validator import DetectionValidator

__all__ = [
    "DetectionEngine",
    "DetectionInput",
    "DetectionPipeline",
    "DetectionResult",
    "DetectionStore",
    "DetectionValidator",
    "FlawPrediction",
    "build_detection_input",
    "build_detection_input_from_features",
]
