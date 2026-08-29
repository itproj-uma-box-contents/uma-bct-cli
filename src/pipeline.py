"""Detector/transcriber-independent orchestration for one shelf image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .domain import BoundingBox, Box, BoxTranscription, InputImage
from .imaging import load_input_image
from .ports import (
    BoxCropper,
    BoxDetector,
    FullImageTextTranscriber,
    TextTranscriber,
)
from .transcriptions import deduplicate, observations_from_response


@dataclass(frozen=True)
class ImageResult:
    """The boxes and transcriptions produced for one input image."""

    input_image: InputImage
    detector: str
    transcriber: str
    boxes: tuple[Box, ...]
    transcriptions: tuple[BoxTranscription, ...]
    warnings: tuple[str, ...] = ()

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
        value: dict[str, object] = {
            "source_image": str(self.input_image.file_path),
            "detector": self.detector,
            "transcriber": self.transcriber,
            "boxes": box_values,
        }
        if self.warnings:
            value["processing_warnings"] = list(self.warnings)
        return value


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
        crop_boxes = self.cropper.create_boxes(image, retained, self.work_directory)
        observations = []
        warnings: list[str] = []
        successful_requests = 0
        for crop_index, box in enumerate(crop_boxes, start=1):
            print(
                f"Crop {crop_index}/{len(crop_boxes)}: "
                f"bbox={box.bounding_box.as_xywh()}, hint={box.position}"
            )
            try:
                returned = self.transcriber.transcribe(box)
            except Exception as error:  # Continue after one failed crop request.
                message = " ".join(str(error).split()) or error.__class__.__name__
                warnings.append(
                    f"Crop {crop_index} {box.bounding_box.as_xywh()} failed: "
                    f"{message[:240]}"
                )
                print(f"\nCrop {crop_index} failed; continuing.")
                continue
            successful_requests += 1
            print(f"\nCrop {crop_index} returned {len(returned)} box(es).")
            observations.extend(
                observations_from_response(box, returned, crop_index)
            )

        if crop_boxes and successful_requests == 0 and isinstance(
            self.transcriber, FullImageTextTranscriber
        ):
            warnings.append(
                "All crop requests failed; retried the original full image once."
            )
            print("All crop requests failed. Retrying the original full image once:")
            full_box = Box(
                box_id=f"{image.file_path.stem}:full-image",
                source_image=image,
                bounding_box=BoundingBox((0, 0), (image.width, image.height)),
                image_file_path=image.file_path,
                position="middle-middle",
                detector_confidence=0.0,
            )
            try:
                returned = self.transcriber.transcribe_full_image(full_box)
            except Exception as error:
                message = " ".join(str(error).split()) or error.__class__.__name__
                warnings.append(f"Full-image retry failed: {message[:240]}")
            else:
                observations.extend(
                    observations_from_response(full_box, returned, 0)
                )

        consolidated = deduplicate(observations)
        return ImageResult(
            input_image=image,
            detector=self.detector.name,
            transcriber=self.transcriber.name,
            boxes=tuple(item.box for item in consolidated),
            transcriptions=tuple(item.transcription for item in consolidated),
            warnings=tuple(warnings),
        )
