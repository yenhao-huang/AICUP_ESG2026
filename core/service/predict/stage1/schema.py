"""Stage 1 prediction CSV schema."""

STAGE = "stage1"
TARGET_FIELD = "promise_status"
ALLOWED_LABELS = ("Yes", "No")
REQUIRED_INPUT_COLUMNS = ("id", TARGET_FIELD)
OUTPUT_COLUMNS = (
    "id",
    TARGET_FIELD,
    "score_yes",
    "score_no",
    "model_family",
    "mode",
    "finetune_path",
    "prompt_id",
    "prompt_path",
    "run_id",
    "source",
    "raw_prediction",
)
DEFAULT_VALUES = {column: "" for column in OUTPUT_COLUMNS}
REASON_COLUMN = "raw_prediction"
SOURCE_COLUMN = None
DEFAULT_KEYWORD_CONFIG = "configs/keyword_analysis_postprocess/stage1.json"
