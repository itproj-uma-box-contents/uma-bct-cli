"""The OpenCV, Qwen, and Ollama adapters."""

from .ollama_client import OllamaVisionModel
from .opencv_detector import OpenCVBoxDetector
from .qwen_detector import QwenBoxDetector
from .qwen_transcriber import QwenTextTranscriber

__all__ = ["OllamaVisionModel", "OpenCVBoxDetector", "QwenBoxDetector", "QwenTextTranscriber"]
