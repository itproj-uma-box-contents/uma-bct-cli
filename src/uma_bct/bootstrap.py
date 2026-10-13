"""Build the detector and transcriber chosen by the CLI.

Qwen shares one Ollama client, and OpenCV is added as its detection fallback.
The model is checked before any image work starts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .adapters import (
    HostedVisionModel,
    OllamaVisionModel,
    OpenCVBoxDetector,
    QwenBoxDetector,
    QwenDateDetector,
    QwenTextTranscriber,
    RegexDateDetector,
)
from .adapters.ollama_client import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .adapters.qwen_transcriber import HOSTED_FULL_IMAGE_QUESTION, hosted_crop_question
from .imaging import OpenCVBoxCropper
from .model_config import HostedModelConfig
from .pipeline import BoxTranscriptionPipeline
from .ports import BoxDetector, TextTranscriber, VisionModelClient
from .terminal import print_selected_model

DetectorFactory = Callable[[VisionModelClient], BoxDetector]
TranscriberFactory = Callable[[VisionModelClient], TextTranscriber]

DETECTORS: dict[str, DetectorFactory] = {"opencv": lambda model_client: OpenCVBoxDetector(), "qwen": QwenBoxDetector}

TRANSCRIBERS: dict[str, TranscriberFactory] = {"qwen": QwenTextTranscriber}

def build_pipeline(*, detector_name: str, transcriber_name: str, ai_dates: bool = False, model_name: str,
                   work_directory: Path, minimum_detector_confidence: float,
                   request_timeout_seconds: float | None = None,
                   hosted_model: HostedModelConfig | None = None) -> BoxTranscriptionPipeline:
    """Build the requested detector and transcriber."""

    try:
        detector_factory = DETECTORS[detector_name]
    except KeyError as error:
        raise ValueError(f"Unknown box detector: {detector_name}") from error
    try:
        transcriber_factory = TRANSCRIBERS[transcriber_name]
    except KeyError as error:
        raise ValueError(f"Unknown text transcriber: {transcriber_name}") from error
    model_client = (
        HostedVisionModel(hosted_model, timeout_seconds=request_timeout_seconds or DEFAULT_REQUEST_TIMEOUT_SECONDS)
        if hosted_model is not None
        else OllamaVisionModel(model=model_name, timeout_seconds=request_timeout_seconds)
    )
    model_metadata = model_client.preflight()
    print_selected_model(model_metadata, fallback_model=hosted_model.model if hosted_model is not None else model_name)
    if detector_name == "qwen" and hosted_model is not None:
        detector = QwenBoxDetector(
            model_client, coordinate_order="yxyx" if hosted_model.provider == "gemini" else "xyxy",
            prompt_profile="hosted",
        )
    else:
        detector = detector_factory(model_client)
    if transcriber_name == "qwen" and hosted_model is not None:
        transcriber = QwenTextTranscriber(
            model_client, crop_prompt=hosted_crop_question, full_image_prompt=HOSTED_FULL_IMAGE_QUESTION
        )
    else:
        transcriber = transcriber_factory(model_client)
    fallback_detector = OpenCVBoxDetector() if detector_name == "qwen" else None
    return BoxTranscriptionPipeline(
        detector=detector,
        fallback_detector=fallback_detector,
        cropper=OpenCVBoxCropper(),
        transcriber=transcriber,
        date_detector=QwenDateDetector(model_client) if ai_dates else RegexDateDetector(),
        work_directory=work_directory,
        minimum_detector_confidence=minimum_detector_confidence,
        model_metadata=model_metadata,
    )
