"""Shared processing defaults, catalogue rules, and image constants."""

OUTPUT_COLUMNS = [
    "Unit Type",
    "Unit Number",
    "Description",
    "Dates",
    "Earliest Date",
    "Latest Date",
    "Transcription Confidence (%)",
    "Review Flags",
    "Cropped Box Image Link",
]

# These exact layouts remain readable when recovering interrupted publications.
DATE_OUTPUT_COLUMNS = OUTPUT_COLUMNS.copy()
LEGACY_OUTPUT_COLUMNS = OUTPUT_COLUMNS[:3] + ["Date Range"] + OUTPUT_COLUMNS[6:]
REVIEW_DATE_OUTPUT_COLUMNS = LEGACY_OUTPUT_COLUMNS + OUTPUT_COLUMNS[3:6]

DUPLICATE_FLAG = "Potential Duplicate"
MISSING_BOX_FLAG = "Missing Box"

LOW_CONFIDENCE_FLAG = "Low confidence"

MISSING_IDENTIFIER_FLAG = "Missing identifier"
AMBIGUOUS_IDENTIFIER_FLAG = "Ambiguous identifier"
MULTIPLE_IDENTIFIERS_FLAG = "Multiple identifiers"
MISSING_UNIT_FLAG = "Missing unit"
INVALID_UNIT_FLAG = "Invalid unit"
TRANSCRIPTION_MISMATCH_FLAG = "Transcription mismatch"
FALLBACK_USED_FLAG = "Fallback used"

# Transcription review uses integer percentages; detector scores use 0 to 1.
DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE = 50
DEFAULT_MIN_DETECTOR_CONFIDENCE = 0.60

MIN_AREA_FRACTION = 0.025
MAX_AREA_FRACTION = 0.45
MIN_WIDTH_FRACTION = 0.08
MIN_HEIGHT_FRACTION = 0.18
MIN_ASPECT_RATIO = 0.30
MAX_ASPECT_RATIO = 1.35
CONTAINMENT_THRESHOLD = 0.90
NMS_IOU_THRESHOLD = 0.45
PADDING_FRACTION = 0.08
MINIMUM_IMAGE_PADDING_FRACTION = 0.005

OPENCV_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PILLOW_IMAGE_EXTENSIONS = {".avif", ".bmp", ".gif", ".heic", ".heif"}
RAW_IMAGE_EXTENSIONS = {".dng"}
IMAGE_EXTENSIONS = OPENCV_IMAGE_EXTENSIONS | PILLOW_IMAGE_EXTENSIONS | RAW_IMAGE_EXTENSIONS
