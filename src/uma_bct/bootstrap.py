"""Build the detector and transcriber chosen by the CLI.

Qwen shares one Ollama client, and OpenCV is added as its detection fallback.
The model is checked before any image work starts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .adapters import OllamaVisionModel, OpenCVBoxDetector, QwenBoxDetector, QwenTextTranscriber, RegexDateDetector, QwenDateDetector, EmptyDateDetector
from .imaging import OpenCVBoxCropper
from .pipeline import BoxTranscriptionPipeline
from .ports import BoxDetector, TextTranscriber, DateDetector, VisionModelClient

DetectorFactory = Callable[[VisionModelClient], BoxDetector]
TranscriberFactory = Callable[[VisionModelClient], TextTranscriber]
DateDetectorFactory = Callable[[VisionModelClient], DateDetector]

DETECTORS: dict[str, DetectorFactory] = {"opencv": lambda model_client: OpenCVBoxDetector(), "qwen": QwenBoxDetector}

TRANSCRIBERS: dict[str, TranscriberFactory] = {"qwen": QwenTextTranscriber}

DATE_DETECTORS: dict[str, TranscriberFactory] = {"none": lambda model_client: EmptyDateDetector(), "regex": lambda model_client: RegexDateDetector(), "qwen": QwenDateDetector}


def build_pipeline(*, detector_name: str, transcriber_name: str, date_detector_name: str = "none", model_name: str, work_directory: Path,
                   minimum_detector_confidence: float, request_timeout_seconds: float | None = None) -> BoxTranscriptionPipeline:
    """Build the requested detector and transcriber."""

    try:
        detector_factory = DETECTORS[detector_name]
    except KeyError as error:
        raise ValueError(f"Unknown box detector: {detector_name}") from error
    try:
        transcriber_factory = TRANSCRIBERS[transcriber_name]
    except KeyError as error:
        raise ValueError(f"Unknown text transcriber: {transcriber_name}") from error
    try:
        date_detector_factory = DATE_DETECTORS[date_detector_name]
    except KeyError as error:
        raise ValueError(f"Unknown date detector: {date_detector_name}") from error

    model_client = OllamaVisionModel(model=model_name, timeout_seconds=request_timeout_seconds)
    model_metadata = model_client.preflight()
    detector = detector_factory(model_client)
    fallback_detector = OpenCVBoxDetector() if detector_name == "qwen" else None
    return BoxTranscriptionPipeline(
        detector=detector,
        fallback_detector=fallback_detector,
        cropper=OpenCVBoxCropper(),
        transcriber=transcriber_factory(model_client),
        date_detector=date_detector_factory(model_client),
        work_directory=work_directory,
        minimum_detector_confidence=minimum_detector_confidence,
        model_metadata=model_metadata,
    )
