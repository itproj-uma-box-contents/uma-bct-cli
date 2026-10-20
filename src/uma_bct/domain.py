"""Shared values used throughout the image pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .identity import date_extraction_text
from .util.date_util import Date, DateRange


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
    processing_file_path: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Input-image dimensions must be positive.")

    @property
    def readable_file_path(self) -> Path:
        """Return a normalized working copy when the source needs one."""

        return self.processing_file_path or self.file_path


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
    dates: tuple[Date | DateRange, ...] = field(default=(), kw_only=True)
    # Separate failed requests from ambiguous source dates. Ambiguity needs
    # human review without making otherwise successful processing partial.
    date_extraction_failed: bool = field(default=False, kw_only=True)
    date_needs_review: bool = field(default=False, kw_only=True)
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
            "dates": [date.to_string() for date in self.dates],
            "auxiliary_annotations": list(self.auxiliary_annotations),
            "transcription_confidence": self.confidence,
        }
        if self.needs_review:
            value["needs_review"] = True
        # Older saved readings omit this field; keep successful output small
        # while distinguishing extraction failures from genuinely undated text.
        if self.date_extraction_failed:
            value["date_extraction_failed"] = True
        if self.date_needs_review:
            value["date_needs_review"] = True
        return value

    def get_handwritten_text(self) -> str:
        """Return handwriting in its original reading order."""
        return "\n".join(self.handwritten_lines)

    def get_date_text(self) -> str:
        """Return date-request text without explicit corroborated identity."""
        return date_extraction_text(self.typed_lines, self.handwritten_lines, self.record_identifiers)


@dataclass(frozen=True)
class BoxResult:
    """A box and the text read from it."""

    box: Box
    transcription: BoxTranscription
    verification: dict[str, object] | None = None

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
        if self.verification is not None:
            value["verification"] = self.verification
        return value
