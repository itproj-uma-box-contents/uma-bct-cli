"""Detector/transcriber-independent orchestration for one shelf image."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Literal

from .domain import Box, BoxTranscription, DetectedRegion, InputImage
from .imaging import create_full_image_box, load_input_image
from .ports import (
    BoxCropper,
    BoxDetector,
    FullImageTextTranscriber,
    TextTranscriber,
)
from .transcriptions import ObservedBox, deduplicate, observations_from_response

ImageStatus = Literal["complete", "partial", "failed"]
DETECTOR_PROPOSAL_AUDIT_LIMIT = 100


def warning_text(error: Exception) -> str:
    """Return one bounded, single-line error suitable for a run manifest."""

    printable = "".join(
        character if character.isprintable() else " " for character in str(error)
    )
    message = " ".join(printable.split()) or error.__class__.__name__
    return message if len(message) <= 240 else f"{message[:237]}..."


def detector_label(detector: BoxDetector) -> str:
    name = detector.name.casefold()
    if name == "qwen":
        return "Qwen"
    if name == "opencv":
        return "OpenCV"
    return detector.name


@dataclass(frozen=True)
class ImageResult:
    """The boxes, transcriptions, and provenance for one input image."""

    input_image: InputImage
    detector: str
    transcriber: str
    boxes: tuple[Box, ...]
    transcriptions: tuple[BoxTranscription, ...]
    warnings: tuple[str, ...] = ()
    requested_detector: str | None = None
    actual_detector: str | None = None
    fallback_path: tuple[str, ...] = ()
    status: ImageStatus = "complete"
    model_metadata: Mapping[str, str] = field(default_factory=dict)
    attempts: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("complete", "partial", "failed"):
            raise ValueError(f"Unknown image processing status: {self.status}")

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

        requested_detector = self.requested_detector or self.detector
        actual_detector = self.actual_detector or self.detector
        fallback_path = self.fallback_path or (requested_detector,)
        value: dict[str, object] = {
            "source_image": str(self.input_image.file_path),
            # Retain the original field for callers consuming the v0 prototype.
            "detector": requested_detector,
            "requested_detector": requested_detector,
            "actual_detector": actual_detector,
            "fallback_path": list(fallback_path),
            "transcriber": self.transcriber,
            "status": self.status,
            "boxes": box_values,
        }
        if self.warnings:
            value["processing_warnings"] = list(self.warnings)
        if self.model_metadata:
            value["model"] = dict(self.model_metadata)
        value["attempts"] = [dict(attempt) for attempt in self.attempts]
        return value


@dataclass(frozen=True)
class Localization:
    boxes: tuple[Box, ...]
    actual_detector: str
    fallback_path: tuple[str, ...]
    full_image: bool = False


@dataclass
class BoxTranscriptionPipeline:
    """Link independent detector, cropper, and transcriber implementations."""

    detector: BoxDetector
    cropper: BoxCropper
    transcriber: TextTranscriber
    work_directory: Path
    minimum_detector_confidence: float = 0.60
    fallback_detector: BoxDetector | None = None
    model_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_detector_confidence <= 1.0:
            raise ValueError("Minimum detector confidence must be between 0 and 1.")

    def retained_regions(self, detector: BoxDetector, detected: Sequence[DetectedRegion]) -> list[DetectedRegion]:
        # Qwen localization does not expose meaningful confidence scores. The
        # user-facing detector threshold therefore applies only to OpenCV.
        if detector.name.casefold() != "opencv":
            return list(detected)
        return [
            region
            for region in detected
            if region.confidence >= self.minimum_detector_confidence
        ]

    def detect_and_crop(self, image: InputImage, detector: BoxDetector,
        warnings: list[str], attempts: list[dict[str, object]]) -> tuple[list[Box], str | None]:

        started_at = perf_counter()
        attempt: dict[str, object] = {
            "stage": "detection",
            "detector": detector.name,
        }
        attempts.append(attempt)
        try:
            detected = detector.detect(image)
        except Exception as error:  # A configured fallback is recovery policy.
            message = warning_text(error)
            attempt.update(
                {
                    "status": "failed",
                    "error": message,
                    "duration_seconds": round(perf_counter() - started_at, 6),
                }
            )
            return [], f"detection failed: {message}"

        detector_warnings = tuple(
            warning_text(ValueError(value))
            for value in getattr(detector, "dropped_proposal_warnings", ())
        )
        if detector_warnings:
            warnings.extend(detector_warnings)
            attempt["dropped_proposal_count"] = len(detector_warnings)
            attempt["dropped_proposal_warnings"] = list(detector_warnings)

        proposal_values = [
            {
                "bbox": list(region.bounding_box.as_xywh()),
                "confidence": region.confidence,
            }
            for region in detected[:DETECTOR_PROPOSAL_AUDIT_LIMIT]
        ]
        attempt.update(
            {
                "proposal_count": len(detected),
                "proposals": proposal_values,
                "proposals_omitted": max(0, len(detected) - len(proposal_values)),
            }
        )

        if not detected:
            attempt.update(
                {
                    "status": "empty",
                    "retained_proposal_count": 0,
                    "crop_count": 0,
                    "duration_seconds": round(perf_counter() - started_at, 6),
                }
            )
            return [], "detection was uncertain"

        retained = self.retained_regions(detector, detected)
        confidence_drops = len(detected) - len(retained)
        if detector.name.casefold() == "opencv" and confidence_drops:
            warnings.append(
                f"OpenCV dropped {confidence_drops} of {len(detected)} proposed "
                "regions below the minimum detector confidence "
                f"{self.minimum_detector_confidence:.2f}."
            )
        attempt["retained_proposal_count"] = len(retained)
        if not retained:
            attempt.update(
                {
                    "status": "empty",
                    "crop_count": 0,
                    "duration_seconds": round(perf_counter() - started_at, 6),
                }
            )
            return (
                [],
                "no crop met the minimum detector confidence "
                f"{self.minimum_detector_confidence:.2f}",
            )

        try:
            boxes = self.cropper.create_boxes(
                image,
                retained,
                self.work_directory,
            )
        except Exception as error:  # A configured fallback is recovery policy.
            message = warning_text(error)
            attempt.update(
                {
                    "status": "failed",
                    "error": f"crop creation failed: {message}",
                    "duration_seconds": round(perf_counter() - started_at, 6),
                }
            )
            return [], f"crop creation failed: {message}"

        attempt["crop_count"] = len(boxes)

        if len(boxes) != len(retained):
            warnings.append(
                f"{detector_label(detector)} proposed {len(retained)} regions; "
                f"retained {len(boxes)} after boundary and overlap checks."
            )
        if not boxes:
            attempt.update(
                {
                    "status": "empty",
                    "duration_seconds": round(perf_counter() - started_at, 6),
                }
            )
            return [], "proposed no usable regions after boundary and overlap checks"
        attempt.update(
            {
                "status": "complete",
                "duration_seconds": round(perf_counter() - started_at, 6),
            }
        )
        return boxes, None

    def localize(self, image: InputImage, warnings: list[str], attempts: list[dict[str, object]]) -> Localization:
        route: list[str] = [self.detector.name]
        boxes, failure = self.detect_and_crop(
            image,
            self.detector,
            warnings,
            attempts,
        )
        if boxes:
            return Localization(tuple(boxes), self.detector.name, tuple(route))

        last_detector = self.detector
        fallback = self.fallback_detector
        if fallback is not None and fallback.name != self.detector.name:
            warnings.append(
                f"{detector_label(self.detector)} box localization failed; used "
                f"{detector_label(fallback)} fallback: {failure}"
            )
            print(
                f"{detector_label(self.detector)} localization failed; using "
                f"{detector_label(fallback)} fallback.",
                file=sys.stderr,
            )
            route.append(fallback.name)
            last_detector = fallback
            boxes, failure = self.detect_and_crop(
                image,
                fallback,
                warnings,
                attempts,
            )
            if boxes:
                return Localization(tuple(boxes), fallback.name, tuple(route))

        if not isinstance(self.transcriber, FullImageTextTranscriber):
            warnings.append(
                f"{detector_label(last_detector)} {failure}; the transcriber does "
                "not support full-image recovery."
            )
            return Localization((), last_detector.name, tuple(route))

        warnings.append(
            f"{detector_label(last_detector)} {failure}; processed the full image "
            "instead."
        )
        route.append("full-image")
        try:
            full_box = create_full_image_box(image, self.work_directory)
        except Exception as error:
            warnings.append(
                f"Could not create the full-image review copy: {warning_text(error)}"
            )
            return Localization((), "full-image", tuple(route), full_image=True)
        return Localization(
            (full_box,),
            "full-image",
            tuple(route),
            full_image=True,
        )

    def transcribe_full_image(self, box: Box, warnings: list[str], attempts: list[dict[str, object]],
        *, warning_prefix: str) -> list[ObservedBox]:
        attempt: dict[str, object] = {
            "stage": "transcription",
            "mode": "full-image",
            "box_id": box.box_id,
            "bbox": list(box.bounding_box.as_xywh()),
        }
        attempts.append(attempt)
        if not isinstance(self.transcriber, FullImageTextTranscriber):
            attempt.update(
                {
                    "status": "unavailable",
                    "error": "transcriber does not support full-image recovery",
                    "duration_seconds": 0.0,
                }
            )
            warnings.append(
                f"{warning_prefix} unavailable: the transcriber does not support "
                "full-image recovery."
            )
            return []
        started_at = perf_counter()
        try:
            returned = self.transcriber.transcribe_full_image(box)
        except Exception as error:
            message = warning_text(error)
            attempt.update(
                {
                    "status": "failed",
                    "error": message,
                    "duration_seconds": round(perf_counter() - started_at, 6),
                }
            )
            warnings.append(f"{warning_prefix} failed: {message}")
            return []
        attempt.update(
            {
                "status": "complete" if returned else "empty",
                "record_count": len(returned),
                "records": [item.to_dict() for item in returned],
                "duration_seconds": round(perf_counter() - started_at, 6),
            }
        )
        if not returned:
            warnings.append(f"{warning_prefix} returned no boxes.")
            return []
        return observations_from_response(box, returned, 0)

    def transcribe_crops(self, localization: Localization, warnings: list[str],
        attempts: list[dict[str, object]]) -> tuple[list[ObservedBox], int]:
        """
        Transcribe localized crops and count requests that remain unresolved.
        """

        observations: list[ObservedBox] = []
        unresolved_crops = 0
        if localization.full_image:
            return observations, unresolved_crops

        for crop_index, box in enumerate(localization.boxes, start=1):
            print(
                f"Crop {crop_index}/{len(localization.boxes)}: "
                f"bbox={box.bounding_box.as_xywh()}, hint={box.position}",
                file=sys.stderr,
            )
            try:
                started_at = perf_counter()
                returned = self.transcriber.transcribe(box)
            except Exception as error:  # Continue after one failed crop request.
                message = warning_text(error)
                attempts.append(
                    {
                        "stage": "transcription",
                        "mode": "crop",
                        "box_id": box.box_id,
                        "bbox": list(box.bounding_box.as_xywh()),
                        "status": "failed",
                        "error": message,
                        "duration_seconds": round(
                            perf_counter() - started_at,
                            6,
                        ),
                    }
                )
                warnings.append(
                    f"Crop {crop_index} {box.bounding_box.as_xywh()} failed: {message}"
                )
                unresolved_crops += 1
                print(f"Crop {crop_index} failed; continuing.", file=sys.stderr)
                continue
            attempts.append(
                {
                    "stage": "transcription",
                    "mode": "crop",
                    "box_id": box.box_id,
                    "bbox": list(box.bounding_box.as_xywh()),
                    "status": "complete" if returned else "empty",
                    "record_count": len(returned),
                    "records": [item.to_dict() for item in returned],
                    "duration_seconds": round(
                        perf_counter() - started_at,
                        6,
                    ),
                }
            )
            if not returned:
                warnings.append(
                    f"Crop {crop_index} {box.bounding_box.as_xywh()} returned no boxes."
                )
                unresolved_crops += 1
                print(f"Crop {crop_index} returned no boxes.", file=sys.stderr)
                continue
            print(
                f"Crop {crop_index} returned {len(returned)} box(es).",
                file=sys.stderr,
            )
            observations.extend(observations_from_response(box, returned, crop_index))

        return observations, unresolved_crops

    def recover(self, image: InputImage, localization: Localization,
        observations: list[ObservedBox], unresolved_crops: int, warnings: list[str],
        attempts: list[dict[str, object]]) -> tuple[list[ObservedBox], int, str, list[str]]:
        """
        Use full-image transcription when localization or every crop fails.
        """

        route = list(localization.fallback_path)
        actual_detector = localization.actual_detector
        if localization.full_image and localization.boxes:
            observations = self.transcribe_full_image(
                localization.boxes[0],
                warnings,
                attempts,
                warning_prefix="Full-image transcription",
            )
            unresolved_crops = 0 if observations else 1
            return observations, unresolved_crops, actual_detector, route

        if localization.boxes and not observations:
            if not isinstance(self.transcriber, FullImageTextTranscriber):
                warnings.append(
                    "All crop requests failed or returned no boxes; full-image "
                    "recovery is unavailable because the transcriber does not "
                    "support full-image recovery."
                )
                print(
                    "All crop requests failed or returned no boxes. Full-image "
                    "recovery is unavailable.",
                    file=sys.stderr,
                )
                return observations, unresolved_crops, actual_detector, route

            if "full-image" not in route:
                route.append("full-image")
            actual_detector = "full-image"
            print(
                "All crop requests failed or returned no boxes. Attempting the "
                "original full image once.",
                file=sys.stderr,
            )
            try:
                full_box = create_full_image_box(image, self.work_directory)
            except Exception as error:
                warnings.append(
                    "All crop requests failed or returned no boxes; attempted "
                    "full-image recovery but could not create the review copy: "
                    f"{warning_text(error)}"
                )
                return observations, unresolved_crops, actual_detector, route

            warnings.append(
                "All crop requests failed or returned no boxes; retried the "
                "original full image once."
            )
            observations = self.transcribe_full_image(
                full_box,
                warnings,
                attempts,
                warning_prefix="Full-image retry",
            )
            if observations and unresolved_crops == 1 and len(observations) == 1:
                # Only one crop and one record can be associated unambiguously.
                unresolved_crops = 0
            elif observations and unresolved_crops:
                warnings.append(
                    "Full-image retry records could not be associated "
                    "one-to-one with failed crops; "
                    f"{unresolved_crops} remain unresolved."
                )

        return observations, unresolved_crops, actual_detector, route

    @staticmethod
    def consolidate(observations: list[ObservedBox]) -> list[ObservedBox]:
        """Merge overlapping observations without merging physical boxes."""

        return deduplicate(observations)

    @staticmethod
    def determine_status(consolidated: Sequence[ObservedBox], unresolved_crops: int,
        attempts: Sequence[Mapping[str, object]]) -> ImageStatus:
        """Classify the image after all recovery and consolidation stages."""

        has_dropped_proposals = any(
            int(attempt.get("dropped_proposal_count", 0)) > 0
            for attempt in attempts
            if attempt.get("stage") == "detection"
        )
        if not consolidated:
            return "failed"
        if unresolved_crops or has_dropped_proposals:
            return "partial"
        return "complete"

    def process(self, image_path: str | Path) -> ImageResult:
        image = load_input_image(image_path)
        warnings: list[str] = []
        attempts: list[dict[str, object]] = []

        localization = self.localize(image, warnings, attempts)
        observations, unresolved_crops = self.transcribe_crops(
            localization,
            warnings,
            attempts,
        )
        observations, unresolved_crops, actual_detector, route = self.recover(
            image,
            localization,
            observations,
            unresolved_crops,
            warnings,
            attempts,
        )
        consolidated = self.consolidate(observations)
        status = self.determine_status(consolidated, unresolved_crops, attempts)

        return ImageResult(
            input_image=image,
            detector=self.detector.name,
            transcriber=self.transcriber.name,
            boxes=tuple(item.box for item in consolidated),
            transcriptions=tuple(item.transcription for item in consolidated),
            warnings=tuple(warnings),
            requested_detector=self.detector.name,
            actual_detector=actual_detector,
            fallback_path=tuple(route),
            status=status,
            model_metadata=self.model_metadata,
            attempts=tuple(attempts),
        )
