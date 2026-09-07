"""Shared OpenCV loading, region validation, and full-resolution cropping."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    CONTAINMENT_THRESHOLD,
    MINIMUM_IMAGE_PADDING_FRACTION,
    NMS_IOU_THRESHOLD,
    PADDING_FRACTION,
)
from .domain import BoundingBox, Box, DetectedRegion, InputImage


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Could not read image: {path}")
    return image


def load_input_image(path: str | Path) -> InputImage:
    file_path = Path(path)
    image = read_image(file_path)
    height, width = image.shape[:2]
    return InputImage(file_path=file_path, width=width, height=height)


def intersection_area(first: BoundingBox, second: BoundingBox) -> int:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.bottom_right[0], second.bottom_right[0])
    bottom = min(first.bottom_right[1], second.bottom_right[1])
    return max(0, right - left) * max(0, bottom - top)


def iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection = intersection_area(first, second)
    if not intersection:
        return 0.0
    first_area = first.width * first.height
    second_area = second.width * second.height
    return intersection / float(first_area + second_area - intersection)


def suppress_overlaps(regions: Sequence[DetectedRegion]) -> list[DetectedRegion]:
    retained: list[DetectedRegion] = []
    for region in sorted(regions, key=lambda item: item.confidence, reverse=True):
        region_area = region.bounding_box.width * region.bounding_box.height
        should_suppress = False
        for kept in retained:
            intersection = intersection_area(region.bounding_box, kept.bounding_box)
            kept_area = kept.bounding_box.width * kept.bounding_box.height
            if max(intersection / region_area, intersection / kept_area) >= (
                CONTAINMENT_THRESHOLD
            ):
                should_suppress = True
                break
            if iou(region.bounding_box, kept.bounding_box) >= NMS_IOU_THRESHOLD:
                should_suppress = True
                break
        if not should_suppress:
            retained.append(region)
    return retained


def sort_in_rows(regions: Sequence[DetectedRegion]) -> list[DetectedRegion]:
    if not regions:
        return []

    median_height = float(np.median([item.bounding_box.height for item in regions]))
    row_tolerance = max(1.0, median_height * 0.35)
    rows: list[list[DetectedRegion]] = []

    for region in sorted(
        regions,
        key=lambda item: item.bounding_box.y + item.bounding_box.height / 2.0,
    ):
        center_y = region.bounding_box.y + region.bounding_box.height / 2.0
        matching_row: list[DetectedRegion] | None = None
        for row in rows:
            row_center = sum(
                item.bounding_box.y + item.bounding_box.height / 2.0 for item in row
            ) / len(row)
            if abs(center_y - row_center) <= row_tolerance:
                matching_row = row
                break
        if matching_row is None:
            rows.append([region])
        else:
            matching_row.append(region)

    rows.sort(
        key=lambda row: (
            sum(item.bounding_box.y + item.bounding_box.height / 2.0 for item in row)
            / len(row)
        )
    )
    return [
        region
        for row in rows
        for region in sorted(row, key=lambda item: item.bounding_box.x)
    ]


def position_hint(box: BoundingBox, image_width: int, image_height: int) -> str:
    column = min(2, int(((box.x + box.width / 2.0) / image_width) * 3))
    row = min(2, int(((box.y + box.height / 2.0) / image_height) * 3))
    return f"{('top', 'middle', 'bottom')[row]}-{('left', 'middle', 'right')[column]}"


def clipped_box(
    box: BoundingBox, image_width: int, image_height: int
) -> BoundingBox | None:
    x1 = max(0, min(image_width, box.top_left[0]))
    y1 = max(0, min(image_height, box.top_left[1]))
    x2 = max(0, min(image_width, box.bottom_right[0]))
    y2 = max(0, min(image_height, box.bottom_right[1]))
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox((x1, y1), (x2, y2))


def image_key(path: Path) -> str:
    """Return a readable, collision-safe identifier for one source image."""

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")
    safe_stem = safe_stem or "image"
    extension = re.sub(r"[^A-Za-z0-9]+", "", path.suffix.casefold())[:16] or "image"
    # Input discovery is intentionally one directory deep, so the filename is
    # the batch-relative path.  Avoid hashing an absolute path: IDs should stay
    # stable when the same batch is moved to another managed workstation.
    normalized_path = path.name
    path_digest = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:10]
    maximum_component_bytes = 120
    stem_budget = maximum_component_bytes - len(extension) - len(path_digest) - 2
    safe_stem = safe_stem[:stem_budget].rstrip("-._") or "image"
    return f"{safe_stem}-{extension}-{path_digest}"


def create_full_image_box(image: InputImage, output_directory: Path) -> Box:
    """Save a reviewable full-image model input under the current run directory."""

    source_key = image_key(image.file_path)
    image_directory = output_directory / source_key
    image_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    image_directory.chmod(0o700)
    image_path = image_directory / "full_image.png"
    if not cv2.imwrite(str(image_path), read_image(image.file_path)):
        raise OSError(f"Could not write full-image review copy: {image_path}")
    image_path.chmod(0o600)
    bounding_box = BoundingBox((0, 0), (image.width, image.height))
    return Box(
        box_id=f"{source_key}:full-image",
        source_image=image,
        bounding_box=bounding_box,
        image_file_path=image_path,
        position="middle-middle",
        detector_confidence=0.0,
    )


class OpenCVBoxCropper:
    """Create clean PNG crops from regions proposed by any detector."""

    def create_boxes(
        self,
        image: InputImage,
        regions: Sequence[DetectedRegion],
        output_directory: Path,
    ) -> list[Box]:
        original = read_image(image.file_path)
        valid_regions: list[DetectedRegion] = []
        for region in regions:
            clipped = clipped_box(region.bounding_box, image.width, image.height)
            if clipped:
                valid_regions.append(DetectedRegion(clipped, region.confidence))

        ordered = sort_in_rows(suppress_overlaps(valid_regions))
        source_key = image_key(image.file_path)
        crop_directory = output_directory / source_key
        crop_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        crop_directory.chmod(0o700)
        boxes: list[Box] = []
        minimum_padding = math.ceil(
            min(image.width, image.height) * MINIMUM_IMAGE_PADDING_FRACTION
        )

        for index, region in enumerate(ordered, start=1):
            box = region.bounding_box
            padding_x = max(round(box.width * PADDING_FRACTION), minimum_padding)
            padding_y = max(round(box.height * PADDING_FRACTION), minimum_padding)
            x1 = max(0, box.x - padding_x)
            y1 = max(0, box.y - padding_y)
            x2 = min(image.width, box.bottom_right[0] + padding_x)
            y2 = min(image.height, box.bottom_right[1] + padding_y)
            padded = BoundingBox((x1, y1), (x2, y2))
            crop_path = crop_directory / f"box_{index:03d}.png"
            crop = original[y1:y2, x1:x2].copy()
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Could not write box crop: {crop_path}")
            crop_path.chmod(0o600)
            boxes.append(
                Box(
                    box_id=f"{source_key}:{index:03d}",
                    source_image=image,
                    bounding_box=padded,
                    image_file_path=crop_path,
                    position=position_hint(padded, image.width, image.height),
                    detector_confidence=region.confidence,
                )
            )
        return boxes
