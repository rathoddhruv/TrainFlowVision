# FOLDER STRUCTURE

The ML component of TrainFlowVision follows a strict and highly organized folder structure to ensure clear separation of concerns, reproducibility, and a pristine deployment environment.

```text
ml/
├── config/                  # Configuration and Path Centralization
│   ├── classes.py           # Class map and ontology management
│   └── paths.py             # Absolute path constants for all scripts
│
├── data/                    # Dataset Management (Ignored by Git)
│   ├── uploads/             # Raw incoming files (not yet reviewed)
│   ├── review_queue/        # Files pending human review
│   ├── reviewed/            # Human-verified annotations
│   │   ├── images/
│   │   └── labels/
│   ├── skipped/             # Ignored or suppressed images
│   ├── training/            # Active YOLO dataset for the next epoch
│   │   ├── images/
│   │   └── labels/
│   └── temp/                # Extracts, caches, and working directories
│
├── docs/                    # ML Specific Documentation
│   ├── ACTIVE_LEARNING_FLOW.md
│   ├── FOLDER_STRUCTURE.md
│   └── ML_PIPELINE.md
│
├── models/                  # Versioned Weight Storage
│   ├── base/                # Initial weights (e.g., yolov8n.pt)
│   ├── current/             # The active model (best.pt) used by the API
│   └── history/             # Archived iterations from previous active learning cycles
│
├── runs/                    # Ephemeral Ultralytics Training Outputs (Ignored by Git)
│   └── detect/
│       └── train/           # The active training run folder
│
├── scripts/                 # Core Execution Logic
│   ├── active_learning_pipeline.py # Training orchestrator
│   ├── boost_merge_labels.py       # Prepares the `training/` dataset
│   ├── import_yolo_dataset_from_zip.py # Initial bootstrap logic
│   └── manual_review.py            # Local CV2 labeling UI (legacy/fallback)
│
└── utils/                   # Specialized helper functions
    ├── cleanup_dataset_folders.py
    └── fix_non_normalized_labels.py
```

### Git Policy
By default, all `data/`, `models/current/`, `models/history/`, and `runs/` are `.gitignore`d to prevent bloat. Only `README.md` markers remain to preserve folder scaffolding.
