"""Current catalogue transcription and export column contracts."""

OUTPUT_COLUMNS = [
    "Node",
    "Title",
    "Description",
    "Start Date",
    "End Date",
    "Confidence Score",
    "Flags",
    "Image",
]

DUPLICATE_FLAG = "Potential Duplicate"

LOW_CONFIDENCE_FLAG = "Low confidence"
LOW_CONFIDENCE_THRESHOLD = 50

MISSING_UNIT_FLAG = "Missing unit"

MIN_AREA_FRACTION = 0.025
MAX_AREA_FRACTION = 0.45
MIN_WIDTH_FRACTION = 0.08
MIN_HEIGHT_FRACTION = 0.18
MIN_ASPECT_RATIO = 0.30
MAX_ASPECT_RATIO = 1.35
CONTAINMENT_THRESHOLD = 0.90
NMS_IOU_THRESHOLD = 0.45
PADDING_FRACTION = 0.05

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

SUPPORTED_OUTPUT_EXTENSIONS = {".csv", ".xlsx"}