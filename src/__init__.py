"""University of Melbourne Archives box transcription pipeline."""

from .domain import (
    BoundingBox,
    Box,
    BoxText,
    BoxTranscription,
    DetectedRegion,
    InputImage,
    Label,
)
from .duplicates import check_duplicate_rows
from .pipeline import BoxTranscriptionPipeline, ImageResult

__all__ = [
    "BoundingBox",
    "Box",
    "BoxText",
    "BoxTranscription",
    "BoxTranscriptionPipeline",
    "check_duplicate_rows",
    "DetectedRegion",
    "ImageResult",
    "InputImage",
    "Label",
]
