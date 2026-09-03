"""Small interfaces that make detection and transcription interchangeable."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from .domain import Box, BoxTranscription, DetectedRegion, InputImage


class BoxDetector(Protocol):
    """Find box regions without creating crops or reading their text."""

    name: str

    def detect(self, image: InputImage) -> list[DetectedRegion]: ...


class BoxCropper(Protocol):
    """Turn detector-neutral regions into clean box crops."""

    def create_boxes(
        self,
        image: InputImage,
        regions: Sequence[DetectedRegion],
        output_directory: Path,
    ) -> list[Box]: ...


class TextTranscriber(Protocol):
    """Transcribe a generic box without knowing which detector produced it."""

    name: str

    def transcribe(self, box: Box) -> list[BoxTranscription]: ...


@runtime_checkable
class FullImageTextTranscriber(Protocol):
    """Optional recovery capability for transcribers that support full images."""

    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]: ...


class VisionModelClient(Protocol):
    """Minimal structured vision-model transport used by Qwen adapters."""

    def query(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
    ) -> Any: ...
