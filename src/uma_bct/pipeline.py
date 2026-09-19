"""Process one shelf image from detection through transcription.

Detection can fall back to OpenCV and then the full image. One bad crop becomes
a warning instead of stopping the whole image.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .domain import Box, BoxResult, DetectedRegion, InputImage
from .imaging import create_full_image_box, load_input_image
from .ports import BoxCropper, BoxDetector, FullImageTextTranscriber, TextTranscriber
from .transcriptions import ObservedBox, deduplicate, observations_from_response

ImageStatus = Literal["complete", "partial", "failed"]


def warning_text(value: Exception | str) -> str:
    """Make a short one-line warning for logs and JSON."""

    printable = "".join(character if character.isprintable() else " " for character in str(value))
    fallback = value.__class__.__name__ if isinstance(value, Exception) else "Warning"
    message = " ".join(printable.split()) or fallback
    return message if len(message) <= 240 else f"{message[:237]}..."


# Keep the image result behavior together.
@dataclass(frozen=True)
class ImageResult:
    input_image: InputImage
    requested_detector: str
    actual_detector: str
    fallback_path: tuple[str, ...]
    transcriber: str
    status: ImageStatus
    records: tuple[BoxResult, ...]
    warnings: tuple[str, ...] = ()
    model_metadata: Mapping[str, str] = field(default_factory=dict)

    # Convert this value to a dictionary.
    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "source_image": str(self.input_image.file_path),
            "requested_detector": self.requested_detector,
            "actual_detector": self.actual_detector,
            "fallback_path": list(self.fallback_path),
            "transcriber": self.transcriber,
            "status": self.status,
            "boxes": [record.to_dict() for record in self.records],
        }
        if self.warnings:
            value["processing_warnings"] = list(self.warnings)
        if self.model_metadata:
            value["model"] = dict(self.model_metadata)
        return value


# Keep the localization behavior together.
@dataclass(frozen=True)
class Localization:
    boxes: tuple[Box, ...]
    actual_detector: str
    route: tuple[str, ...]
    full_image: bool = False
    incomplete: bool = False


# Keep the box transcription pipeline behavior together.
@dataclass
class BoxTranscriptionPipeline:
    detector: BoxDetector
    cropper: BoxCropper
    transcriber: TextTranscriber
    work_directory: Path
    minimum_detector_confidence: float = 0.60
    fallback_detector: BoxDetector | None = None
    model_metadata: dict[str, str] = field(default_factory=dict)

    # Validate the newly created value.
    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_detector_confidence <= 1.0:
            raise ValueError("Minimum detector confidence must be between 0 and 1.")

    # Handle the retained regions step.
    def retained_regions(self, detector: BoxDetector, regions: Sequence[DetectedRegion]) -> list[DetectedRegion]:
        if detector.name.casefold() != "opencv":
            return list(regions)
        return [region for region in regions if region.confidence >= self.minimum_detector_confidence]

    # Handle the detect and crop step.
    def detect_and_crop(self, image: InputImage, detector: BoxDetector, warnings: list[str]) -> tuple[list[Box], str | None, bool]:
        try:
            detected = detector.detect(image)
        except Exception as error:
            return [], f"detection failed: {warning_text(error)}", False

        dropped_warnings = tuple(warning_text(value) for value in getattr(detector, "dropped_proposal_warnings", ()))
        warnings.extend(dropped_warnings)
        retained = self.retained_regions(detector, detected)
        dropped_for_confidence = len(detected) - len(retained)
        if dropped_for_confidence:
            warnings.append(
                f"OpenCV dropped {dropped_for_confidence} of {len(detected)} "
                f"regions below {self.minimum_detector_confidence:.2f} confidence."
            )
        if not retained:
            reason = "detection returned no boxes" if not detected else "no box met the detector confidence threshold"
            return [], reason, bool(dropped_warnings or dropped_for_confidence)

        try:
            boxes = self.cropper.create_boxes(image, retained, self.work_directory)
        except Exception as error:
            return [], f"crop creation failed: {warning_text(error)}", False
        if not boxes:
            return [], "detection returned no usable crop", False
        if len(boxes) != len(retained):
            warnings.append(f"{detector.name} proposed {len(retained)} regions; {len(boxes)} remained after crop checks.")
        return boxes, None, bool(dropped_warnings or dropped_for_confidence)


    # Handle the full image localization step.
    def full_image_localization(self, image: InputImage, warnings: list[str], detector: BoxDetector,
                                failure: str | None, route: list[str], incomplete: bool) -> Localization:

        if not isinstance(self.transcriber, FullImageTextTranscriber):
            warnings.append(f"{detector.name} localization failed ({failure}); full-image recovery is unavailable.")
            return Localization((), detector.name, tuple(route), incomplete=True)

        warnings.append(f"{detector.name} localization failed ({failure}); processed the full image instead.")
        route.append("full-image")
        try:
            full_box = create_full_image_box(image, self.work_directory)
        except Exception as error:
            warnings.append(f"Could not create the full-image review copy: {warning_text(error)}")
            return Localization((), "full-image", tuple(route), full_image=True, incomplete=True)
        return Localization((full_box,), "full-image", tuple(route), full_image=True, incomplete=incomplete)

    # Handle the localize step.
    def localize(self, image: InputImage, warnings: list[str]) -> Localization:
        route = [self.detector.name]
        boxes, failure, incomplete = self.detect_and_crop(image, self.detector, warnings)
        if boxes:
            return Localization(tuple(boxes), self.detector.name, tuple(route), incomplete=incomplete)

        last_detector = self.detector
        fallback = self.fallback_detector
        if fallback is not None and fallback.name != self.detector.name:
            warnings.append(f"{self.detector.name} localization failed; used {fallback.name}: {failure}")
            print(f"{self.detector.name} localization failed; using {fallback.name}.", file=sys.stderr)
            route.append(fallback.name)
            last_detector = fallback
            boxes, failure, fallback_incomplete = self.detect_and_crop(image, fallback, warnings)
            incomplete |= fallback_incomplete
            if boxes:
                return Localization(tuple(boxes), fallback.name, tuple(route), incomplete=incomplete)
        return self.full_image_localization(image, warnings, last_detector, failure, route, incomplete)

    # Transcribe the full image box.
    def transcribe_full_image_box(self, box: Box, warnings: list[str], label: str) -> list[ObservedBox]:
        if not isinstance(self.transcriber, FullImageTextTranscriber):
            warnings.append(f"{label} is unavailable.")
            return []
        try:
            returned = self.transcriber.transcribe_full_image(box)
        except Exception as error:
            warnings.append(f"{label} failed: {warning_text(error)}")
            return []
        if not returned:
            warnings.append(f"{label} returned no boxes.")
            return []
        return observations_from_response(box, returned, 0)

    # Transcribe the crops.
    def transcribe_crops(self, boxes: Sequence[Box], warnings: list[str]) -> tuple[list[ObservedBox], int]:
        observations: list[ObservedBox] = []
        unresolved = 0
        for index, box in enumerate(boxes, start=1):
            print(f"Crop {index}/{len(boxes)}: bbox={box.bounding_box.as_xywh()}, hint={box.position}", file=sys.stderr)
            try:
                returned = self.transcriber.transcribe(box)
            except Exception as error:
                warnings.append(f"Crop {index} {box.bounding_box.as_xywh()} failed: {warning_text(error)}")
                unresolved += 1
                continue
            if not returned:
                warnings.append(f"Crop {index} {box.bounding_box.as_xywh()} returned no boxes.")
                unresolved += 1
                continue
            observations.extend(observations_from_response(box, returned, index))
        return observations, unresolved


    # Recover the failed crops.
    def recover_failed_crops(self, image: InputImage, localization: Localization, observations: list[ObservedBox],
                             unresolved: int, warnings: list[str]) -> tuple[list[ObservedBox], int, str, tuple[str, ...]]:

        route = list(localization.route)
        actual_detector = localization.actual_detector

        # Retry the full image only when every crop failed.
        if localization.full_image or not localization.boxes or observations:
            return observations, unresolved, actual_detector, tuple(route)
        if not isinstance(self.transcriber, FullImageTextTranscriber):
            warnings.append("Full-image retry is unavailable.")
            return observations, unresolved, actual_detector, tuple(route)

        print("All crop requests failed or returned no boxes; trying the full image once.", file=sys.stderr)
        try:
            full_box = create_full_image_box(image, self.work_directory)
        except Exception as error:
            warnings.append(f"Full-image retry could not create a review copy: {warning_text(error)}")
            return observations, unresolved, actual_detector, tuple(route)
        if "full-image" not in route:
            route.append("full-image")
        recovered = self.transcribe_full_image_box(full_box, warnings, "Full-image retry")
        if len(localization.boxes) == 1 and len(recovered) == 1:
            unresolved = 0
        return recovered, unresolved, "full-image", tuple(route)

    # Process the current input.
    def process(self, image_path: str | Path) -> ImageResult:
        image = load_input_image(image_path)
        warnings: list[str] = []
        localization = self.localize(image, warnings)
        if localization.full_image and localization.boxes:
            observations = self.transcribe_full_image_box(localization.boxes[0], warnings, "Full-image transcription")
            unresolved = 0 if observations else 1
        else:
            observations, unresolved = self.transcribe_crops(localization.boxes, warnings)
        observations, unresolved, actual_detector, route = self.recover_failed_crops(
            image, localization, observations, unresolved, warnings
        )

        consolidated = deduplicate(observations)
        status: ImageStatus
        if not consolidated:
            status = "failed"
        elif unresolved or localization.incomplete:
            status = "partial"
        else:
            status = "complete"
        return ImageResult(
            input_image=image,
            requested_detector=self.detector.name,
            actual_detector=actual_detector,
            fallback_path=route,
            transcriber=self.transcriber.name,
            status=status,
            records=tuple(BoxResult(item.box, item.transcription) for item in consolidated),
            warnings=tuple(warnings),
            model_metadata=self.model_metadata,
        )
