"""Geometry-based archive-box detection implemented with OpenCV.

This is the offline, model-free fallback detector: it finds box-like
rectangles purely from edges and lines, with no vision model involved.
Detection runs on a downscaled copy (``DETECTION_MAX_DIMENSION``) for speed,
and ``detect_regions`` maps candidate boxes back to full-resolution source
coordinates by scaling each axis independently against the *actual* resized
dimensions (which are rounded independently per axis) rather than a single
nominal scale factor, to match a known-good reference implementation exactly.
``candidate_confidence`` is a hand-tuned weighted score (area, aspect ratio,
size, rectangularity, polygon-approximation shape, and Hough-line edge
support) rather than a learned model; ``line_support`` in particular checks
whether each of the four candidate sides has a nearby straight line, which
disambiguates a real box edge from an accidental rectangle formed by shadows
or shelf clutter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..constants import (
    MAX_AREA_FRACTION,
    MAX_ASPECT_RATIO,
    MIN_AREA_FRACTION,
    MIN_ASPECT_RATIO,
    MIN_HEIGHT_FRACTION,
    MIN_WIDTH_FRACTION,
)
from ..domain import BoundingBox, DetectedRegion, InputImage
from ..imaging import read_image, sort_in_rows, suppress_overlaps

DETECTION_MAX_DIMENSION = 1400
INTERNAL_CONFIDENCE_FLOOR = 0.45


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def range_score(value: float, minimum: float, preferred_minimum: float, preferred_maximum: float, maximum: float) -> float:
    """Score a value with a flat preferred range and tapered outer ranges."""

    if value < minimum or value > maximum:
        return 0.0
    if preferred_minimum <= value <= preferred_maximum:
        return 1.0
    if value < preferred_minimum:
        return (value - minimum) / (preferred_minimum - minimum)
    return (maximum - value) / (maximum - preferred_maximum)


def odd_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, round(value))
    return size if size % 2 else size + 1


def plausible_bbox(box: BoundingBox, image_width: int, image_height: int) -> bool:
    """Apply conservative geometry only to OpenCV-generated proposals."""

    area_fraction = (box.width * box.height) / float(image_width * image_height)
    aspect_ratio = box.width / float(box.height)
    return (
        MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION
        and box.width >= image_width * MIN_WIDTH_FRACTION
        and box.height >= image_height * MIN_HEIGHT_FRACTION
        and MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO
    )


def line_segments(edges: np.ndarray) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    height, width = edges.shape
    short_side = min(height, width)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, int(short_side * 0.04)),
        minLineLength=max(20, int(short_side * 0.08)),
        maxLineGap=max(8, int(short_side * 0.02)),
    )
    horizontal: list[tuple[int, ...]] = []
    vertical: list[tuple[int, ...]] = []
    if lines is None:
        return horizontal, vertical

    for packed_line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in packed_line)
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle <= 12 or angle >= 168:
            horizontal.append((x1, y1, x2, y2))
        elif 78 <= angle <= 102:
            vertical.append((x1, y1, x2, y2))
    return horizontal, vertical


def line_support(
    bbox: tuple[int, int, int, int], horizontal: list[tuple[int, ...]], vertical: list[tuple[int, ...]]
) -> float:
    """Estimate how many candidate sides have nearby straight-line support."""

    x, y, width, height = bbox
    horizontal_tolerance = max(5, int(height * 0.09))
    vertical_tolerance = max(5, int(width * 0.09))
    top = bottom = left = right = False

    for x1, y1, x2, y2 in horizontal:
        line_length = abs(x2 - x1)
        overlaps_x = min(max(x1, x2), x + width) - max(min(x1, x2), x)
        if line_length < width * 0.25 or overlaps_x < width * 0.20:
            continue
        line_y = (y1 + y2) / 2.0
        top = top or abs(line_y - y) <= horizontal_tolerance
        bottom = bottom or abs(line_y - (y + height)) <= horizontal_tolerance

    for x1, y1, x2, y2 in vertical:
        line_length = abs(y2 - y1)
        overlaps_y = min(max(y1, y2), y + height) - max(min(y1, y2), y)
        if line_length < height * 0.25 or overlaps_y < height * 0.20:
            continue
        line_x = (x1 + x2) / 2.0
        left = left or abs(line_x - x) <= vertical_tolerance
        right = right or abs(line_x - (x + width)) <= vertical_tolerance
    return sum((top, bottom, left, right)) / 4.0


def candidate_confidence(
    contour: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    horizontal_lines: list[tuple[int, ...]],
    vertical_lines: list[tuple[int, ...]],
) -> float:
    _, _, width, height = bbox
    image_height, image_width = image_shape
    image_area = float(image_width * image_height)
    box_area = float(width * height)
    area_fraction = box_area / image_area
    aspect_ratio = width / float(height)
    width_fraction = width / float(image_width)
    height_fraction = height / float(image_height)

    contour_area = abs(float(cv2.contourArea(contour)))
    rectangularity = contour_area / box_area if box_area else 0.0
    perimeter = float(cv2.arcLength(contour, True))
    polygon: Any = cv2.approxPolyDP(contour, 0.02 * perimeter, True) if perimeter else []
    area_score = range_score(area_fraction, MIN_AREA_FRACTION, 0.05, 0.25, MAX_AREA_FRACTION)
    aspect_score = range_score(aspect_ratio, MIN_ASPECT_RATIO, 0.42, 0.90, MAX_ASPECT_RATIO)
    size_score = (clamp(width_fraction / 0.16) + clamp(height_fraction / 0.36)) / 2.0
    rectangularity_score = clamp((rectangularity - 0.18) / 0.62)
    polygon_score = 1.0 if 4 <= len(polygon) <= 8 else 0.35
    lines_score = line_support(bbox, horizontal_lines, vertical_lines)

    confidence = (
        0.22 * area_score
        + 0.22 * aspect_score
        + 0.10 * size_score
        + 0.18 * rectangularity_score
        + 0.13 * polygon_score
        + 0.15 * lines_score
    )
    if aspect_ratio > 0.90:
        confidence *= aspect_score**2
    return round(clamp(confidence), 3)


def detection_evidence(detection_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return source edges and the joined rectangle mask used for contours."""

    height, width = detection_image.shape[:2]
    gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.55 * median))
    upper = int(min(255, 1.25 * median))
    if upper - lower < 30:
        lower = max(0, upper - 30)
    edges = cv2.Canny(blurred, lower, upper)

    short_side = min(height, width)
    base_size = odd_size(short_side * 0.004)
    horizontal_size = odd_size(width * 0.012)
    vertical_size = odd_size(height * 0.012)
    base_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (base_size, base_size))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, base_kernel, iterations=2)
    horizontal = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, horizontal_kernel)
    vertical = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, vertical_kernel)
    joined = cv2.bitwise_or(horizontal, vertical)
    joined = cv2.dilate(joined, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return edges, joined


def extract_candidates(detection_image: np.ndarray) -> list[DetectedRegion]:
    """Isolated OpenCV logic that can be replaced from the reference project."""

    height, width = detection_image.shape[:2]
    edges, joined = detection_evidence(detection_image)

    horizontal_lines, vertical_lines = line_segments(edges)
    contours, _ = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[DetectedRegion] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue
        bounding_box = BoundingBox((x, y), (x + box_width, y + box_height))
        if not plausible_bbox(bounding_box, width, height):
            continue

        bbox = (x, y, box_width, box_height)
        confidence = candidate_confidence(contour, bbox, (height, width), horizontal_lines, vertical_lines)
        if confidence >= INTERNAL_CONFIDENCE_FLOOR:
            candidates.append(DetectedRegion(bounding_box, confidence))
    return candidates


def detect_regions(original: np.ndarray) -> list[DetectedRegion]:
    """Return OpenCV regions in full-resolution source coordinates."""

    if original is None or original.size == 0:
        raise ValueError("The original image is empty.")
    original_height, original_width = original.shape[:2]
    longest_side = max(original_height, original_width)
    scale = min(1.0, DETECTION_MAX_DIMENSION / float(longest_side))
    if scale < 1.0:
        detection_image = cv2.resize(
            original,
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_image = original.copy()

    detection_height, detection_width = detection_image.shape[:2]
    candidates = sort_in_rows(suppress_overlaps(extract_candidates(detection_image)))
    regions: list[DetectedRegion] = []
    for candidate in candidates:
        box = candidate.bounding_box
        # Use the real resized dimensions so rounding does not shift the box.
        left = box.x * original_width // detection_width
        top = box.y * original_height // detection_height
        right = (box.bottom_right[0] * original_width + detection_width - 1) // detection_width
        bottom = (box.bottom_right[1] * original_height + detection_height - 1) // detection_height
        regions.append(DetectedRegion(BoundingBox((left, top), (right, bottom)), candidate.confidence))
    return regions


@dataclass(frozen=True)
class OpenCVBoxDetector:
    """Adapter exposing the reference OpenCV detector through BoxDetector."""

    name: str = "opencv"

    def detect(self, image: InputImage) -> list[DetectedRegion]:
        return detect_regions(read_image(image.file_path))
