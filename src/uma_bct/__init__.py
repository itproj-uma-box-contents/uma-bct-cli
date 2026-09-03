"""University of Melbourne Archives box transcription pipeline."""

from .catalogue import map_catalogue_record
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
from .export import catalogue_dataframe, export_catalogue
from .pipeline import BoxTranscriptionPipeline, ImageResult
from .validation import check_low_confidence_rows

__all__ = [
    "BoundingBox",
    "Box",
    "BoxText",
    "BoxTranscription",
    "BoxTranscriptionPipeline",
    "catalogue_dataframe",
    "check_duplicate_rows",
    "check_low_confidence_rows",
    "DetectedRegion",
    "ImageResult",
    "InputImage",
    "Label",
    "export_catalogue",
    "map_catalogue_record",
]
