# settings.py
from pathlib import Path
from ml.config.paths import (
    ML_DIR,
    REVIEW_QUEUE_DIR,
    ML_TEMP_DIR,
    TRAINING_IMAGES_DIR,
    TRAINING_LABELS_DIR
)

# project root
ROOT_DIR = Path(__file__).resolve().parents[1]

# paths
UPLOAD_DIR = REVIEW_QUEUE_DIR
LABEL_STUDIO_DIR = ML_TEMP_DIR / "label_studio_exports"

# scripts
ML_PIPELINE = ML_DIR / "scripts" / "active_learning_pipeline.py"
IMPORT_ZIP_SCRIPT = ML_DIR / "scripts" / "import_yolo_dataset_from_zip.py"

# optional dataset helpers
YOLO_DATASET_ROOT = ML_TEMP_DIR / "yolo_dataset"
YOLO_DATASET_IMAGES = YOLO_DATASET_ROOT / "images" / "train"
YOLO_DATASET_LABELS = YOLO_DATASET_ROOT / "labels" / "train"
YOLO_MERGED_ROOT = ML_DIR / "data" / "training"
YOLO_MERGED_IMAGES = TRAINING_IMAGES_DIR
YOLO_MERGED_LABELS = TRAINING_LABELS_DIR
