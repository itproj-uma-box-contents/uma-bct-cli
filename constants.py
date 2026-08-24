# Schema produced by image transcription.
TRANSCRIPTION_COLUMNS = [
    "Node ID",
    "Node Type",
    "Node Title",
    "Description",
    "Date Range",
    "Image",
]

# The exported CSV adds review flags to the transcription data.
OUTPUT_COLUMNS = [*TRANSCRIPTION_COLUMNS, "Flags"]
