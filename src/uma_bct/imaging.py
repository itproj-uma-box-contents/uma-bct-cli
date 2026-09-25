"""Read source images and make ordered box crops.

Crops always come from the full-size photo. Overlapping detections are cleaned
up, sorted into shelf order, and saved with stable file names.
"""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .constants import (
    CONTAINMENT_THRESHOLD,
    MINIMUM_IMAGE_PADDING_FRACTION,
    NMS_IOU_THRESHOLD,
    OPENCV_IMAGE_EXTENSIONS,
    PADDING_FRACTION,
    PILLOW_IMAGE_EXTENSIONS,
    RAW_IMAGE_EXTENSIONS,
)
from .domain import BoundingBox, Box, DetectedRegion, InputImage
from .path_safety import check_path


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_input_image(image: InputImage) -> np.ndarray:
    """Read the private normalized copy when one was prepared."""

    return read_image(image.readable_file_path)


def pillow_image(path: Path) -> np.ndarray:
    """Decode the main still image, applying orientation and removing alpha."""

    if path.suffix.casefold() in {".heic", ".heif"}:
        # The plugin opens the container's primary image and applies its HEIF
        # transforms. Thumbnails, depth maps, and auxiliary images are unused.
        register_heif_opener(thumbnails=False, depth_images=False, aux_images=False)
    try:
        with Image.open(path) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                rgb = background.convert("RGB")
            else:
                rgb = oriented.convert("RGB")
            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError(f"Could not read image: {path}: {error}") from error


def raw_image(path: Path) -> np.ndarray:
    """Develop a full-resolution RAW source into oriented 8-bit sRGB."""

    try:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                half_size=False,
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
            )
    except Exception as error:
        raise ValueError(f"Could not read image: {path}: {error}") from error
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size == 0:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)


def decode_new_image(path: Path) -> np.ndarray:
    extension = path.suffix.casefold()
    if extension in PILLOW_IMAGE_EXTENSIONS:
        return pillow_image(path)
    if extension in RAW_IMAGE_EXTENSIONS:
        return raw_image(path)
    raise ValueError(f"Unsupported image extension: {path.suffix or '<none>'}")


@contextmanager
def prepare_input_image(path: str | Path) -> Iterator[InputImage]:
    """Yield one source image with a private normalized path when required."""

    file_path = Path(path)
    if file_path.suffix.casefold() in OPENCV_IMAGE_EXTENSIONS:
        yield load_input_image(file_path)
        return

    # The source path remains the catalogue identity. Formats OpenCV cannot
    # read directly use a short-lived PNG without carrying source metadata on.
    pixels = decode_new_image(file_path)
    height, width = pixels.shape[:2]
    with tempfile.TemporaryDirectory(prefix="uma-bct-input-") as directory:
        temporary_directory = Path(directory)
        temporary_directory.chmod(0o700)
        processing_path = temporary_directory / "source.png"
        if not cv2.imwrite(str(processing_path), pixels):
            raise OSError(f"Could not create normalized image: {file_path}")
        del pixels
        processing_path.chmod(0o600)
        yield InputImage(
            file_path=file_path,
            width=width,
            height=height,
            processing_file_path=processing_path,
        )

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
    # Containment catches duplicate proposals of very different sizes; IoU
    # catches proposals whose outlines mostly coincide.
    for region in sorted(regions, key=lambda item: item.confidence, reverse=True):
        region_area = region.bounding_box.width * region.bounding_box.height
        should_suppress = False
        for kept in retained:
            intersection = intersection_area(region.bounding_box, kept.bounding_box)
            kept_area = kept.bounding_box.width * kept.bounding_box.height
            if max(intersection / region_area, intersection / kept_area) >= (CONTAINMENT_THRESHOLD):
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

    # Compare vertical centres using the typical box height so uneven shelves
    # still read top to bottom, then left to right within each row.
    median_height = float(np.median([item.bounding_box.height for item in regions]))
    row_tolerance = max(1.0, median_height * 0.35)
    rows: list[list[DetectedRegion]] = []

    for region in sorted(regions, key=lambda item: item.bounding_box.y + item.bounding_box.height / 2.0):
        center_y = region.bounding_box.y + region.bounding_box.height / 2.0
        matching_row: list[DetectedRegion] | None = None
        for row in rows:
            row_center = sum(item.bounding_box.y + item.bounding_box.height / 2.0 for item in row) / len(row)
            if abs(center_y - row_center) <= row_tolerance:
                matching_row = row
                break
        if matching_row is None:
            rows.append([region])
        else:
            matching_row.append(region)

    rows.sort(key=lambda row: sum(item.bounding_box.y + item.bounding_box.height / 2.0 for item in row) / len(row))
    return [region for row in rows for region in sorted(row, key=lambda item: item.bounding_box.x)]


def position_hint(box: BoundingBox, image_width: int, image_height: int) -> str:
    column = min(2, int(((box.x + box.width / 2.0) / image_width) * 3))
    row = min(2, int(((box.y + box.height / 2.0) / image_height) * 3))
    return f"{('top', 'middle', 'bottom')[row]}-{('left', 'middle', 'right')[column]}"


def clipped_box(box: BoundingBox, image_width: int, image_height: int) -> BoundingBox | None:
    x1 = max(0, min(image_width, box.top_left[0]))
    y1 = max(0, min(image_height, box.top_left[1]))
    x2 = max(0, min(image_width, box.bottom_right[0]))
    y2 = max(0, min(image_height, box.bottom_right[1]))
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox((x1, y1), (x2, y2))


def image_key(path: Path) -> str:
    """Make a stable folder name for one source image."""

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")
    safe_stem = safe_stem or "image"
    extension = re.sub(r"[^A-Za-z0-9]+", "", path.suffix.casefold())[:16] or "image"
    # Use the filename so IDs stay stable when a batch moves to another computer.
    normalized_path = path.name
    path_digest = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:10]
    maximum_component_bytes = 120
    stem_budget = maximum_component_bytes - len(extension) - len(path_digest) - 2
    safe_stem = safe_stem[:stem_budget].rstrip("-._") or "image"
    return f"{safe_stem}-{extension}-{path_digest}"


def create_full_image_box(image: InputImage, output_directory: Path) -> Box:
    """Save a full-image copy for review or fallback."""

    source_key = image_key(image.file_path)
    image_directory = output_directory / source_key
    # Check again here in case the output path changed during model work.
    check_path(image_directory, "Full-image review directory")
    image_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    check_path(image_directory, "Full-image review directory")
    image_directory.chmod(0o700)
    image_path = image_directory / "full_image.png"
    check_path(image_path, "Full-image review copy")
    if not cv2.imwrite(str(image_path), read_input_image(image)):
        raise OSError(f"Could not write full-image review copy: {image_path}")
    check_path(image_path, "Full-image review copy")
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
    """Create clean PNG crops from detected boxes."""

    def create_boxes(self, image: InputImage, regions: Sequence[DetectedRegion], output_directory: Path) -> list[Box]:
        original = read_input_image(image)
        valid_regions: list[DetectedRegion] = []
        for region in regions:
            clipped = clipped_box(region.bounding_box, image.width, image.height)
            if clipped:
                valid_regions.append(DetectedRegion(clipped, region.confidence))

        # Assign stable crop numbers only after clipping, overlap removal, and
        # visual ordering; the CSV will inherit this physical reading order.
        ordered = sort_in_rows(suppress_overlaps(valid_regions))
        source_key = image_key(image.file_path)
        crop_directory = output_directory / source_key
        # Check again here in case the output path changed during model work.
        check_path(crop_directory, "Review-crop directory")
        crop_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        check_path(crop_directory, "Review-crop directory")
        crop_directory.chmod(0o700)
        boxes: list[Box] = []
        minimum_padding = math.ceil(min(image.width, image.height) * MINIMUM_IMAGE_PADDING_FRACTION)

        for index, region in enumerate(ordered, start=1):
            box = region.bounding_box
            # A little surrounding shelf gives a reviewer context around the
            # label, including boxes near an image edge.
            padding_x = max(round(box.width * PADDING_FRACTION), minimum_padding)
            padding_y = max(round(box.height * PADDING_FRACTION), minimum_padding)
            x1 = max(0, box.x - padding_x)
            y1 = max(0, box.y - padding_y)
            x2 = min(image.width, box.bottom_right[0] + padding_x)
            y2 = min(image.height, box.bottom_right[1] + padding_y)
            padded = BoundingBox((x1, y1), (x2, y2))
            crop_path = crop_directory / f"box_{index:03d}.png"
            crop = original[y1:y2, x1:x2].copy()
            check_path(crop_path, "Review crop")
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Could not write box crop: {crop_path}")
            check_path(crop_path, "Review crop")
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
