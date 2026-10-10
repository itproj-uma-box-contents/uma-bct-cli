"""Build the detector and transcriber chosen by the CLI.

Qwen shares one Ollama client, and OpenCV is added as its detection fallback.
The model is checked before any image work starts.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from .adapters import OllamaVisionModel, OpenCVBoxDetector, QwenBoxDetector, QwenTextTranscriber
from .imaging import OpenCVBoxCropper
from .pipeline import BoxTranscriptionPipeline
from .ports import BoxDetector, TextTranscriber, VisionModelClient

DetectorFactory = Callable[[VisionModelClient], BoxDetector]
TranscriberFactory = Callable[[VisionModelClient], TextTranscriber]

DETECTORS: dict[str, DetectorFactory] = {"opencv": lambda model_client: OpenCVBoxDetector(), "qwen": QwenBoxDetector}

TRANSCRIBERS: dict[str, TranscriberFactory] = {"qwen": QwenTextTranscriber}


def build_pipeline(*, detector_name: str, transcriber_name: str, model_name: str, work_directory: Path,
                   minimum_detector_confidence: float, request_timeout_seconds: float | None = None) -> BoxTranscriptionPipeline:
    """Build the requested detector and transcriber."""

    workers = os.getenv("UMA_BCT_DEMO_TRANSCRIPTION_WORKERS", "1")
    if workers not in {"1", "2"}:
        raise ValueError("UMA_BCT_DEMO_TRANSCRIPTION_WORKERS must be 1 or 2.")
    try:
        detector_factory = DETECTORS[detector_name]
    except KeyError as error:
        raise ValueError(f"Unknown box detector: {detector_name}") from error
    try:
        transcriber_factory = TRANSCRIBERS[transcriber_name]
    except KeyError as error:
        raise ValueError(f"Unknown text transcriber: {transcriber_name}") from error

    model_client = OllamaVisionModel(model=model_name, timeout_seconds=request_timeout_seconds)
    if workers == "2" and model_client.effective_host != "http://ollama:11434":
        raise ValueError("Parallel demo transcription requires its prepared private Compose runners.")
    model_metadata = model_client.preflight()
    parallel_transcribers = ()
    if workers == "2":
        second = OllamaVisionModel(model=model_name, timeout_seconds=request_timeout_seconds,
                                   host="http://ollama-parallel:11434")
        second_metadata = second.preflight()
        if any(second_metadata.get(key) != model_metadata.get(key) for key in ("model", "model_digest")):
            raise RuntimeError("Parallel runner model identity differs from the primary runner.")
        model_metadata = {**model_metadata, "demo_transcription_workers": "2",
                          "demo_parallel_ollama_host": second.effective_host}
        parallel_transcribers = (transcriber_factory(second),)
    detector = detector_factory(model_client)
    fallback_detector = OpenCVBoxDetector() if detector_name == "qwen" else None
    return BoxTranscriptionPipeline(
        detector=detector,
        fallback_detector=fallback_detector,
        cropper=OpenCVBoxCropper(),
        transcriber=transcriber_factory(model_client),
        work_directory=work_directory,
        minimum_detector_confidence=minimum_detector_confidence,
        model_metadata=model_metadata,
        parallel_transcribers=parallel_transcribers,
    )
