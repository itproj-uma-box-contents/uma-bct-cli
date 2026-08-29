"""Detector/transcriber-independent orchestration for one shelf image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .domain import Box, BoxTranscription, InputImage
from .imaging import load_input_image
from .ports import BoxCropper, BoxDetector, TextTranscriber


@dataclass(frozen=True)
class ImageResult:
    """The boxes and transcriptions produced for one input image."""

    input_image: InputImage
    detector: str
    transcriber: str
    boxes: tuple[Box, ...]
    transcriptions: tuple[BoxTranscription, ...]

    def to_dict(self) -> dict[str, object]:
        transcription_by_id = {
            item.box_id: item.to_dict() for item in self.transcriptions
        }
        box_values: list[dict[str, object]] = []
        for box in self.boxes:
            value: dict[str, object] = {
                "box_id": box.box_id,
                "bbox": list(box.bounding_box.as_xywh()),
                "image": str(box.image_file_path),
                "detector_confidence": box.detector_confidence,
            }
            value.update(transcription_by_id.get(box.box_id, {}))
            box_values.append(value)
        return {
            "source_image": str(self.input_image.file_path),
            "detector": self.detector,
            "transcriber": self.transcriber,
            "boxes": box_values,
        }


@dataclass
class BoxTranscriptionPipeline:
    """Link independent detector, cropper, and transcriber implementations."""

    detector: BoxDetector
    cropper: BoxCropper
    transcriber: TextTranscriber
    work_directory: Path
    minimum_detector_confidence: float = 0.60

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_detector_confidence <= 1.0:
            raise ValueError("Minimum detector confidence must be between 0 and 1.")

    def process(self, image_path: str | Path) -> ImageResult:
        image = load_input_image(image_path)
        detected = self.detector.detect(image)
        retained = [
            region
            for region in detected
            if region.confidence >= self.minimum_detector_confidence
        ]
        boxes = self.cropper.create_boxes(image, retained, self.work_directory)
        transcriptions = [self.transcriber.transcribe(box) for box in boxes]
        return ImageResult(
            input_image=image,
            detector=self.detector.name,
            transcriber=self.transcriber.name,
            boxes=tuple(boxes),
            transcriptions=tuple(transcriptions),
        )
