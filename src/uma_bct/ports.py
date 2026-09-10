"""Small interfaces shared by the pipeline and adapters.

Detection, cropping, and transcription can be swapped without changing the
pipeline. Full-image transcription stays optional.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .domain import Box, BoxTranscription, DetectedRegion, InputImage


class BoxDetector(Protocol):
    """Find boxes without reading their text."""

    name: str

    def detect(self, image: InputImage) -> list[DetectedRegion]: ...


class BoxCropper(Protocol):
    """Turn detected boxes into clean crops."""

    def create_boxes(self, image: InputImage, regions: Sequence[DetectedRegion], output_directory: Path) -> list[Box]: ...


class TextTranscriber(Protocol):
    """Read the text from one box."""

    name: str

    def transcribe(self, box: Box) -> list[BoxTranscription]: ...


@runtime_checkable
class FullImageTextTranscriber(Protocol):
    """Read a full image when crop processing fails."""

    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]: ...


class VisionModelClient(Protocol):
    """Send one image request to the vision model."""

    def query(
        self, image_path: Path, prompt: str, schema: dict[str, Any], *, validator: Callable[[Any], object] | None = None
    ) -> Any: ...
