"""Stage 2 prediction CSV schema."""

STAGE = "stage2"
TARGET_FIELD = "evidence_status"
ALLOWED_LABELS = ("Yes", "No", "N/A")
REQUIRED_INPUT_COLUMNS = ("id", TARGET_FIELD)
OUTPUT_COLUMNS = (
    "id",
    TARGET_FIELD,
    "evidence_status_raw",
    "filter_passed",
    "prediction_source",
    "postprocess_reason",
)
DEFAULT_VALUES = {column: "" for column in OUTPUT_COLUMNS}
REASON_COLUMN = "postprocess_reason"
SOURCE_COLUMN = "prediction_source"
DEFAULT_KEYWORD_CONFIG = "configs/keyword_analysis_postprocess/stage2.json"
