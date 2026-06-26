"""Stage 3 prediction CSV schema."""

STAGE = "stage3"
TARGET_FIELD = "evidence_quality"
ALLOWED_LABELS = ("Clear", "Not Clear", "Misleading", "N/A")
REQUIRED_INPUT_COLUMNS = ("id", TARGET_FIELD)
OUTPUT_COLUMNS = (
    "id",
    TARGET_FIELD,
    "evidence_quality_raw",
    "evidence_quality_source",
    "evidence_quality_reason",
)
DEFAULT_VALUES = {column: "" for column in OUTPUT_COLUMNS}
REASON_COLUMN = "evidence_quality_reason"
SOURCE_COLUMN = "evidence_quality_source"
DEFAULT_KEYWORD_CONFIG = "configs/keyword_analysis_postprocess/stage3.json"
