from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BE_DIR = ROOT_DIR / "BE"
FE_DIR = ROOT_DIR / "FE"
ML_DIR = ROOT_DIR / "ml"

ML_DATA_DIR = ML_DIR / "data"
UPLOADS_DIR = ML_DATA_DIR / "uploads"
REVIEW_QUEUE_DIR = ML_DATA_DIR / "review_queue"
REVIEWED_IMAGES_DIR = ML_DATA_DIR / "reviewed" / "images"
REVIEWED_LABELS_DIR = ML_DATA_DIR / "reviewed" / "labels"
TRAINING_IMAGES_DIR = ML_DATA_DIR / "training" / "images"
TRAINING_LABELS_DIR = ML_DATA_DIR / "training" / "labels"
SKIPPED_DIR = ML_DATA_DIR / "skipped"
ML_TEMP_DIR = ML_DATA_DIR / "temp"

BASE_MODELS_DIR = ML_DIR / "models" / "base"
CURRENT_MODEL_DIR = ML_DIR / "models" / "current"
MODEL_HISTORY_DIR = ML_DIR / "models" / "history"
RUNS_DIR = ML_DIR / "runs"
