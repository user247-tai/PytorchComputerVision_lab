from .base import PytorchVisionLab
from .classification import Classification
from .detection import Detection
from .instance_segmentation import InstanceSegmentation
from .semantic_segmentation import BACKGROUND_INDEX, INPUT_SIZE, NUM_CLASSES, SemanticSegmentation

__all__ = [
    "PytorchVisionLab",
    "Classification",
    "Detection",
    "InstanceSegmentation",
    "SemanticSegmentation",
    "BACKGROUND_INDEX",
    "INPUT_SIZE",
    "NUM_CLASSES",
]
