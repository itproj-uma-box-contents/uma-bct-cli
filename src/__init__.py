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
from .catalogue import map_catalogue_record
from .export import catalogue_dataframe, export_catalogue
from .pipeline import BoxTranscriptionPipeline, ImageResult

__all__ = [
    "BoundingBox",
    "Box",
    "BoxText",
    "BoxTranscription",
    "BoxTranscriptionPipeline",
    "catalogue_dataframe",
    "check_duplicate_rows",
    "DetectedRegion",
    "ImageResult",
    "InputImage",
    "Label",
    "export_catalogue",
    "map_catalogue_record",
]
