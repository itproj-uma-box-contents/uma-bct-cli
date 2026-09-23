"""The OpenCV, Qwen, and Ollama adapters."""

from .ollama_client import OllamaVisionModel
from .opencv_detector import OpenCVBoxDetector
from .qwen_detector import QwenBoxDetector
from .qwen_transcriber import QwenTextTranscriber
from .regex_date_detector import RegexDateDetector
from .qwen_date_detector import QwenDateDetector
from .empty_date_detector import EmptyDateDetector

__all__ = ["OllamaVisionModel", "OpenCVBoxDetector", "QwenBoxDetector", "QwenTextTranscriber", "RegexDateDetector", "QwenDateDetector", "EmptyDateDetector"]
