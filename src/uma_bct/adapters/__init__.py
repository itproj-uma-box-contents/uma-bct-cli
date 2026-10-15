"""The OpenCV, Qwen, Ollama, and hosted adapters."""

from .empty_date_detector import EmptyDateDetector
from .hosted_client import HostedVisionModel
from .ollama_client import OllamaVisionModel
from .opencv_detector import OpenCVBoxDetector
from .qwen_date_detector import QwenDateDetector
from .qwen_detector import QwenBoxDetector
from .qwen_transcriber import QwenTextTranscriber
from .regex_date_detector import RegexDateDetector

__all__ = [
    "EmptyDateDetector",
    "HostedVisionModel",
    "OllamaVisionModel",
    "OpenCVBoxDetector",
    "QwenBoxDetector",
    "QwenDateDetector",
    "QwenTextTranscriber",
    "RegexDateDetector",
]
