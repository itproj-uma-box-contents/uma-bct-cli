"""Shared OpenCV loading, region validation, and full-resolution cropping."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .domain import BoundingBox, Box, DetectedRegion, InputImage

MIN_AREA_FRACTION = 0.025
MAX_AREA_FRACTION = 0.45
MIN_WIDTH_FRACTION = 0.08
MIN_HEIGHT_FRACTION = 0.18
MIN_ASPECT_RATIO = 0.30
MAX_ASPECT_RATIO = 1.35
CONTAINMENT_THRESHOLD = 0.90
NMS_IOU_THRESHOLD = 0.45
PADDING_FRACTION = 0.05


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
            intersection = intersection_area(
                region.bounding_box, kept.bounding_box
            )
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

    median_height = float(
        np.median([item.bounding_box.height for item in regions])
    )
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
        key=lambda row: sum(
            item.bounding_box.y + item.bounding_box.height / 2.0 for item in row
        )
        / len(row)
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


def plausible_bbox(box: BoundingBox, image_width: int, image_height: int) -> bool:
    area_fraction = (box.width * box.height) / float(image_width * image_height)
    aspect_ratio = box.width / float(box.height)
    return (
        MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION
        and box.width >= image_width * MIN_WIDTH_FRACTION
        and box.height >= image_height * MIN_HEIGHT_FRACTION
        and MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO
    )


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
            if clipped and plausible_bbox(clipped, image.width, image.height):
                valid_regions.append(DetectedRegion(clipped, region.confidence))

        ordered = sort_in_rows(suppress_overlaps(valid_regions))
        crop_directory = output_directory / image.file_path.stem
        crop_directory.mkdir(parents=True, exist_ok=True)
        boxes: list[Box] = []

        for index, region in enumerate(ordered, start=1):
            box = region.bounding_box
            padding_x = round(box.width * PADDING_FRACTION)
            padding_y = round(box.height * PADDING_FRACTION)
            x1 = max(0, box.x - padding_x)
            y1 = max(0, box.y - padding_y)
            x2 = min(image.width, box.bottom_right[0] + padding_x)
            y2 = min(image.height, box.bottom_right[1] + padding_y)
            padded = BoundingBox((x1, y1), (x2, y2))
            crop_path = crop_directory / f"box_{index:03d}.png"
            crop = original[y1:y2, x1:x2].copy()
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Could not write box crop: {crop_path}")
            boxes.append(
                Box(
                    box_id=f"{image.file_path.stem}:{index:03d}",
                    source_image=image,
                    bounding_box=padded,
                    image_file_path=crop_path,
                    position=position_hint(padded, image.width, image.height),
                    detector_confidence=region.confidence,
                )
            )
        return boxes
