"""The OpenCV, Qwen, and Ollama adapters."""

from .empty_date_detector import EmptyDateDetector
from .ollama_client import OllamaVisionModel
from .opencv_detector import OpenCVBoxDetector
from .qwen_date_detector import QwenDateDetector
from .qwen_detector import QwenBoxDetector
from .qwen_transcriber import QwenTextTranscriber
from .regex_date_detector import RegexDateDetector

__all__ = ["OllamaVisionModel", "OpenCVBoxDetector", "QwenBoxDetector", "QwenTextTranscriber", "RegexDateDetector", "QwenDateDetector", "EmptyDateDetector"]
