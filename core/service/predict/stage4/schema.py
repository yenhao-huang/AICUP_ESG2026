"""Stage 4 prediction CSV schema."""

STAGE = "stage4"
TARGET_FIELD = "verification_timeline"
ALLOWED_LABELS = (
    "already",
    "within_2_years",
    "between_2_and_5_years",
    "more_than_5_years",
    "N/A",
)
REQUIRED_INPUT_COLUMNS = ("id", TARGET_FIELD)
OUTPUT_COLUMNS = (
    "id",
    TARGET_FIELD,
    "stage4_flow",
    "stage1_promise_str",
    "stage4_filtered",
    "stage4_raw_timeline",
    "stage4_postprocess_rule",
    "stage4_error",
)
DEFAULT_VALUES = {column: "" for column in OUTPUT_COLUMNS}
REASON_COLUMN = "stage4_postprocess_rule"
SOURCE_COLUMN = None
DEFAULT_KEYWORD_CONFIG = "configs/keyword_analysis_postprocess/stage4.json"
