"""Small interfaces shared by the pipeline and adapters.

Detection, cropping, and transcription can be swapped without changing the
pipeline. Full-image transcription stays optional.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .domain import Box, BoxTranscription, DetectedRegion, InputImage
from .util.date_util import Date, DateRange


class BoxDetector(Protocol):
    """Find boxes without reading their text."""

    name: str

    # Detect boxes in the current image.
    def detect(self, image: InputImage) -> list[DetectedRegion]: ...


class BoxCropper(Protocol):
    """Turn detected boxes into clean crops."""

    # Create the boxes.
    def create_boxes(self, image: InputImage, regions: Sequence[DetectedRegion], output_directory: Path) -> list[Box]: ...


class TextTranscriber(Protocol):
    """Read the text from one box."""

    name: str

    # Transcribe the current box.
    def transcribe(self, box: Box) -> list[BoxTranscription]: ...


class DateDetector(Protocol):
    """Read the text from one box."""

    name: str

    def extract_dates(self, text: str) -> tuple[Date | DateRange, ...]: ...


@runtime_checkable
class FullImageTextTranscriber(Protocol):
    """Read a full image when crop processing fails."""

    # Transcribe the full image when crops are unavailable.
    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]: ...


@dataclass(frozen=True)
class ModelResponse:
    """A query's parsed value plus the token log-probabilities behind it.

    ``token_logprobs`` is empty unless ``logprobs=True`` was requested and
    returned; treat an empty tuple as "unavailable", not as "confident".
    """

    value: Any
    token_logprobs: tuple[tuple[str, float], ...] = ()


class VisionModelClient(Protocol):
    """Send one image request to the vision model."""

    # Run the model query.
    def query(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
        *,
        validator: Callable[[Any], object] | None = None,
        logprobs: bool = False,
    ) -> ModelResponse: ...
