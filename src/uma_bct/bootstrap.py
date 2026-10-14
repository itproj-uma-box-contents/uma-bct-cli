"""Build the detector and transcriber chosen by the CLI.

Qwen shares one Ollama client, and OpenCV is added as its detection fallback.
The model is checked before any image work starts.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from queue import Queue

from .adapters import OllamaVisionModel, OpenCVBoxDetector, QwenBoxDetector, QwenTextTranscriber
from .imaging import OpenCVBoxCropper
from .pipeline import BoxTranscriptionPipeline
from .ports import BoxDetector, TextTranscriber, VisionModelClient

DetectorFactory = Callable[[VisionModelClient], BoxDetector]
TranscriberFactory = Callable[[VisionModelClient], TextTranscriber]

DETECTORS: dict[str, DetectorFactory] = {"opencv": lambda model_client: OpenCVBoxDetector(), "qwen": QwenBoxDetector}

TRANSCRIBERS: dict[str, TranscriberFactory] = {"qwen": QwenTextTranscriber}


def parallel_preflight(readers: tuple[OllamaVisionModel, ...]) -> list[dict]:
    """Check independent services without making cancellation wait for HTTP timeouts."""
    completed: Queue[tuple[int, dict | BaseException]] = Queue()
    def check(index: int, reader: OllamaVisionModel) -> None:
        try:
            value = reader.preflight()
        except BaseException as error:
            value = error
        completed.put((index, value))
    for index, reader in enumerate(readers):
        threading.Thread(target=check, args=(index, reader), daemon=True).start()
    values = {}
    for _ in readers:
        index, value = completed.get()
        if isinstance(value, BaseException) and not isinstance(value, Exception):
            raise value
        values[index] = value
    metadata = []
    for index in range(len(readers)):
        value = values[index]
        if isinstance(value, Exception):
            raise value
        metadata.append(value)
    return metadata


def build_pipeline(*, detector_name: str, transcriber_name: str, model_name: str, work_directory: Path,
                   minimum_detector_confidence: float, request_timeout_seconds: float | None = None) -> BoxTranscriptionPipeline:
    """Build the requested detector and transcriber."""

    workers = os.getenv("UMA_BCT_DEMO_TRANSCRIPTION_WORKERS", "1")
    if workers not in {"1", "2", "4", "6", "8"}:
        raise ValueError("UMA_BCT_DEMO_TRANSCRIPTION_WORKERS must be 1, 2, 4, 6 or 8.")
    try:
        detector_factory = DETECTORS[detector_name]
    except KeyError as error:
        raise ValueError(f"Unknown box detector: {detector_name}") from error
    try:
        transcriber_factory = TRANSCRIBERS[transcriber_name]
    except KeyError as error:
        raise ValueError(f"Unknown text transcriber: {transcriber_name}") from error

    model_client = OllamaVisionModel(model=model_name, timeout_seconds=request_timeout_seconds)
    if workers != "1" and model_client.effective_host != "http://ollama:11434":
        raise ValueError("Parallel demo transcription requires its prepared private Compose runners.")
    model_metadata = model_client.preflight()
    parallel_transcribers = ()
    if workers != "1":
        configured_hosts = os.getenv("UMA_BCT_DEMO_READER_HOSTS")
        hosts = (json.loads(configured_hosts) if configured_hosts else
                 [f"http://ollama-parallel{'' if index == 2 else f'-{index}'}:11434"
                  for index in range(2, int(workers) + 1)])
        if not isinstance(hosts, list) or len(hosts) != int(workers) - 1 or not all(isinstance(host, str) for host in hosts):
            raise ValueError("Prepared demo reader endpoints must match the declared worker count.")
        readers = tuple(OllamaVisionModel(model=model_name, timeout_seconds=request_timeout_seconds,
                        host=host) for host in hosts)
        # Independent services can be checked concurrently without sharing a request slot.
        metadata = parallel_preflight(readers)
        if any(item.get(key) != model_metadata.get(key) for item in metadata for key in ("model", "model_digest")):
            raise RuntimeError("Parallel runner model identity differs from the primary runner.")
        model_metadata = {**model_metadata, "demo_transcription_workers": workers,
                          "demo_parallel_ollama_host": ",".join(reader.effective_host for reader in readers)}
        parallel_transcribers = tuple(transcriber_factory(reader) for reader in readers)
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
