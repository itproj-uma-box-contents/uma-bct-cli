"""The OpenCV, Qwen, Ollama, and hosted adapters."""

from .hosted_client import HostedVisionModel
from .ollama_client import OllamaVisionModel
from .opencv_detector import OpenCVBoxDetector
from .qwen_detector import QwenBoxDetector
from .qwen_transcriber import QwenTextTranscriber

__all__ = ["HostedVisionModel", "OllamaVisionModel", "OpenCVBoxDetector", "QwenBoxDetector", "QwenTextTranscriber"]
