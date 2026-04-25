"""
File: active_learning_pipeline.py
Purpose: Runs the active learning training flow for TrainFlowVision.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from ml.config.paths import (
    MODEL_HISTORY_DIR, 
    RUNS_DIR, 
    TRAINING_IMAGES_DIR, 
    TRAINING_LABELS_DIR,
    ML_TEMP_DIR as TEMP_DIR,
    ML_DIR as ML_ROOT
)
from ultralytics import YOLO
from multiprocessing import freeze_support

# Constants
RUNS_DETECT = RUNS_DIR / "detect"
TRAIN_STABLE = RUNS_DETECT / "train"

def get_device():
    return "0" if torch.cuda.is_available() else "cpu"

def _archive_existing_train():
    if not TRAIN_STABLE.exists():
        return None
    MODEL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = MODEL_HISTORY_DIR / f"train_{ts}"
    shutil.move(str(TRAIN_STABLE), str(dst))
    print(f"Archived previous train to: {dst}")
    return dst

def _manifest_append(event: str, extra: dict):
    try:
        RUNS_DETECT.mkdir(parents=True, exist_ok=True)
        mf = RUNS_DETECT / "manifest.json"
        data = []
        if mf.exists():
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
            except:
                data = []
        rec = {"event": event, "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")}
        rec.update(extra or {})
        mf.write_text(json.dumps([*data, rec], indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Manifest append failed: {e}")

def _sync_yaml(yaml_path: Path, data_path: Path):
    from ml.config.classes import CLASS_FILE
    try:
        names_dict = {}
        if CLASS_FILE.exists():
            lines = CLASS_FILE.read_text(encoding="utf-8").splitlines()
            names_dict = {i: line.strip() for i, line in enumerate(lines) if line.strip()}
        
        if not names_dict:
            names_dict = {0: "object"}
            
        train_dir = "images/train" if (data_path / "images" / "train").exists() else "images"
        content = [
            f"path: {data_path.as_posix()}",
            f"train: {train_dir}",
            f"val: {train_dir}",
            "names:",
        ]
        for idx, nm in names_dict.items():
            content.append(f"  {idx}: {nm}")
            
        yaml_path.write_text("\n".join(content), encoding="utf-8")
        print(f"Synced {yaml_path.name} to {data_path}")
    except Exception as e:
        print(f"YAML Sync failed: {e}")

def _merge_classes_from_dataset(dataset_root: Path):
    from ml.config.classes import CLASS_FILE
    ds_classes_file = dataset_root / "classes.txt"
    if not ds_classes_file.exists():
        return
    
    try:
        project_classes = []
        if CLASS_FILE.exists():
            project_classes = [line.strip() for line in CLASS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        
        ds_classes = [line.strip() for line in ds_classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        
        new_classes = 0
        for cls in ds_classes:
            if cls not in project_classes:
                project_classes.append(cls)
                new_classes += 1
        
        if new_classes > 0:
            CLASS_FILE.write_text("\n".join(project_classes), encoding="utf-8")
            print(f"Added {new_classes} new classes from imported dataset.")
    except Exception as e:
        print(f"Failed to merge classes: {e}")

def validate_dataset_before_training(yaml_path: str):
    print("\n--- Validating Dataset Before YOLO Training ---")
    import yaml
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        base_path = Path(data['path'])
        train_dir = data.get('train', 'images')
        images_folder = base_path / train_dir
        
        if not images_folder.exists():
            raise FileNotFoundError(f"Missing images folder: {images_folder}")
            
        num_images = sum(1 for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'] for _ in images_folder.glob(ext))
        print(f"Found {num_images} images at {images_folder}")
        
        if num_images == 0:
            raise ValueError("No images found in training set.")

        label_dir = train_dir.replace('images', 'labels')
        labels_folder = base_path / label_dir
        
        if not labels_folder.exists():
             raise FileNotFoundError(f"Missing labels folder: {labels_folder}")
            
        num_labels = len(list(labels_folder.glob('*.txt')))
        print(f"Found {num_labels} labels at {labels_folder}")
        
        print("--- Validation Passed ---\n")
    except Exception as e:
        print(f"\n[CRITICAL] Dataset validation failed: {str(e)}")
        sys.exit(1)

def get_task(model_name: str) -> str:
    name = model_name.lower()
    if "obb" in name: return "obb"
    if "seg" in name: return "segment"
    return "detect"

if __name__ == '__main__':
    freeze_support()
    print("=== STARTING TRAINFLOW PIPELINE ===")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_ROOT)

    YOLO_MERGED_YAML_ABS = str((ML_ROOT / "yolo_merged.yaml").resolve())

    # 1. Detect Initial Dataset
    initial_dataset_root = TEMP_DIR / "yolo_dataset"
    initial_images = initial_dataset_root / "images" / "train"
    initial_labels = initial_dataset_root / "labels" / "train"
    
    has_initial_data = False
    if initial_images.exists() and any(initial_images.glob("*")):
        print("Detected initial Label Studio dataset. Preparing...")
        _merge_classes_from_dataset(initial_dataset_root)
        from ml.utils.fix_non_normalized_labels_logic import normalize_folder
        normalize_folder(initial_images, initial_labels)
        has_initial_data = True

    # 2. CLI Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    # 3. Optional Cleanup
    if args.clean:
        print("Cleaning dataset folders...")
        subprocess.run([sys.executable, "ml/utils/cleanup_dataset_folders.py"], check=True)

    # 4. Manual Review
    if not args.no_interactive:
        print("Launching manual review...")
        try:
            subprocess.run([sys.executable, "ml/scripts/manual_review.py"], check=True)
        except Exception as e:
            print(f"Manual review failed: {e}")
            sys.exit(1)

    # 5. Merge Data
    print("Merging datasets...")
    subprocess.run([sys.executable, "ml/scripts/boost_merge_labels.py"], check=True)

    if has_initial_data:
        USED_PATH = TEMP_DIR / "yolo_dataset_used"
        if USED_PATH.exists(): shutil.rmtree(USED_PATH)
        shutil.move(str(initial_dataset_root), str(USED_PATH))
        print(f"Initial dataset moved to {USED_PATH}")

    # 6. Validate
    _sync_yaml(Path(YOLO_MERGED_YAML_ABS), ML_ROOT / "data/training")
    validate_dataset_before_training(YOLO_MERGED_YAML_ABS)

    # 7. Training
    train_images = list(TRAINING_IMAGES_DIR.glob("*"))
    if not train_images:
        print("No images found for training.")
        sys.exit(1)

    _archive_existing_train()
    
    # Select model
    # Priority: 1. previous best.pt, 2. user arg model
    MODEL_PATH = args.model
    potential_best = RUNS_DETECT / "train" / "weights" / "best.pt"
    if potential_best.exists():
        MODEL_PATH = str(potential_best)
    
    print(f"Training from: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    device = get_device()
    task_type = get_task(str(MODEL_PATH))

    try:
        model.train(
            task=task_type,
            data=YOLO_MERGED_YAML_ABS,
            imgsz=args.imgsz,
            device=device,
            project=str(RUNS_DETECT),
            name="train",
            exist_ok=True,
            epochs=args.epochs,
            lr0=0.005,
            amp=False,
            val=False
        )
        print("SUCCESS: Pipeline completed")
        
        # Manifest
        _manifest_append("train_success", {
            "images": len(train_images),
            "epochs": args.epochs,
            "model": MODEL_PATH
        })
        
    except Exception as e:
        print(f"CRITICAL: Training failed: {e}")
        sys.exit(1)
