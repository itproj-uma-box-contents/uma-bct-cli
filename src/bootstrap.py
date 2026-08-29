"""Composition root: the only module that selects concrete components."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .adapters import (
    OllamaVisionModel,
    OpenCVBoxDetector,
    QwenBoxDetector,
    QwenTextTranscriber,
)
from .imaging import OpenCVBoxCropper
from .pipeline import BoxTranscriptionPipeline
from .ports import BoxDetector, TextTranscriber, VisionModelClient

DetectorFactory = Callable[[VisionModelClient], BoxDetector]
TranscriberFactory = Callable[[VisionModelClient], TextTranscriber]

DETECTORS: dict[str, DetectorFactory] = {
    "opencv": lambda _client: OpenCVBoxDetector(),
    "qwen": QwenBoxDetector,
}

TRANSCRIBERS: dict[str, TranscriberFactory] = {
    "qwen": QwenTextTranscriber,
}


def build_pipeline(
    *,
    detector_name: str,
    transcriber_name: str,
    model_name: str,
    work_directory: Path,
    minimum_detector_confidence: float,
) -> BoxTranscriptionPipeline:
    """Resolve the requested adapters and inject them into one pipeline."""

    try:
        detector_factory = DETECTORS[detector_name]
    except KeyError as error:
        raise ValueError(f"Unknown box detector: {detector_name}") from error
    try:
        transcriber_factory = TRANSCRIBERS[transcriber_name]
    except KeyError as error:
        raise ValueError(f"Unknown text transcriber: {transcriber_name}") from error

    model_client = OllamaVisionModel(model=model_name)
    return BoxTranscriptionPipeline(
        detector=detector_factory(model_client),
        cropper=OpenCVBoxCropper(),
        transcriber=transcriber_factory(model_client),
        work_directory=work_directory,
        minimum_detector_confidence=minimum_detector_confidence,
    )
