"""Small interfaces that make detection and transcription interchangeable.

These ``Protocol`` classes are the seam between ``pipeline.py`` (which only
depends on these interfaces) and the concrete adapters in ``adapters/``
(which implement them for Qwen and OpenCV). ``bootstrap.py`` is the only
place that wires a concrete adapter to a name; nothing else needs to import
from ``adapters`` directly. ``FullImageTextTranscriber`` is a separate,
optional protocol (checked with ``runtime_checkable``/``isinstance``) rather
than part of ``TextTranscriber``, since only some transcribers can usefully
process a whole photo when box localization fails.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .domain import Box, BoxTranscription, DetectedRegion, InputImage


class BoxDetector(Protocol):
    """Find box regions without creating crops or reading their text."""

    name: str

    def detect(self, image: InputImage) -> list[DetectedRegion]: ...


class BoxCropper(Protocol):
    """Turn detector-neutral regions into clean box crops."""

    def create_boxes(self, image: InputImage, regions: Sequence[DetectedRegion], output_directory: Path) -> list[Box]: ...


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
        self, image_path: Path, prompt: str, schema: dict[str, Any], *, validator: Callable[[Any], object] | None = None
    ) -> Any: ...
