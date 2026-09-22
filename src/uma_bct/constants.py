"""Catalogue rules and output column names."""

OUTPUT_COLUMNS = [
    "Unit Type",
    "Unit Number",
    "Description",
    "Date Range",
    "Transcription Confidence (%)",
    "Review Flags",
    "Cropped Box Image Link",
]

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
