"""Shared values used throughout the image pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from .util.date_util import Date, DateRange
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox:
    """A rectangle in the original image."""

    top_left: tuple[int, int]
    bottom_right: tuple[int, int]

    def __post_init__(self) -> None:
        x1, y1 = self.top_left
        x2, y2 = self.bottom_right
        if min(x1, y1) < 0:
            raise ValueError("Bounding-box coordinates cannot be negative.")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding-box bottom-right must follow top-left.")

    @property
    def x(self) -> int:
        return self.top_left[0]

    @property
    def y(self) -> int:
        return self.top_left[1]

    @property
    def width(self) -> int:
        return self.bottom_right[0] - self.top_left[0]

    @property
    def height(self) -> int:
        return self.bottom_right[1] - self.top_left[1]

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True)
class InputImage:
    """One shelf image supplied to the pipeline."""

    file_path: Path
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Input-image dimensions must be positive.")


@dataclass(frozen=True)
class DetectedRegion:
    """The location of one possible archive box."""

    bounding_box: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detector confidence must be between 0 and 1.")


@dataclass(frozen=True)
class Box:
    """A full-size crop of one box."""

    box_id: str
    source_image: InputImage
    bounding_box: BoundingBox
    image_file_path: Path
    position: str
    detector_confidence: float


@dataclass(frozen=True)
class BoxTranscription:
    """Validated text visible on one physical archive box.

    ``confidence`` comes from the model's own token log-probabilities (see
    ``adapters.qwen_transcriber``), not a value the model reports about itself.
    """

    box_id: str
    position: str
    confidence: int
    record_identifiers: tuple[str, ...] = ()
    unit_number: str = ""
    typed_lines: tuple[str, ...] = ()
    handwritten_lines: tuple[str, ...] = ()
    dates: tuple[Date | DateRange, ...] = ()
    auxiliary_annotations: tuple[str, ...] = ()
    needs_review: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 100:
            raise ValueError("Transcription confidence must be between 0 and 100.")

    def to_dict(self) -> dict[str, object]:
        """Build the small version saved in the run file."""

        value: dict[str, object] = {
            "box_id": self.box_id,
            "position": self.position,
            "record_identifiers": list(self.record_identifiers),
            "unit": self.unit_number,
            "typed": list(self.typed_lines),
            "handwritten": list(self.handwritten_lines),
            "dates": list([date.to_string() for date in self.dates]),
            "auxiliary_annotations": list(self.auxiliary_annotations),
            "transcription_confidence": self.confidence,
        }
        if self.needs_review:
            value["needs_review"] = True
        return value

    def get_handwritten_text(self) -> str:
            """Gets the handwritten text as a single string."""
            return "\n".join(self.handwritten_lines)


@dataclass(frozen=True)
class BoxResult:
    """A box and the text read from it."""

    box: Box
    transcription: BoxTranscription

    def __post_init__(self) -> None:
        if self.box.box_id != self.transcription.box_id:
            raise ValueError("Box and transcription IDs must match.")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "box_id": self.box.box_id,
            "bbox": list(self.box.bounding_box.as_xywh()),
            "image": str(self.box.image_file_path),
            "detector_confidence": self.box.detector_confidence,
        }
        value.update(self.transcription.to_dict())
        return value
