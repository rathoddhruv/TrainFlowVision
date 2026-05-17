# System Architecture

TrainFlowVision is a full-stack, end-to-end active learning vision platform.

## Components

1. **Frontend (Angular 17)**
   - Located in `/FE`
   - Uses NgRx for state management
   - Uses Tailwind CSS and Angular Material for styling
   - Communicates with the backend via REST API

2. **Backend (FastAPI)**
   - Located in `/BE`
   - Serves as the API gateway for the frontend
   - Manages uploaded files and stages them for the ML pipeline
   - Provides real-time inference using the currently loaded YOLO model
   - Orchestrates background training tasks

3. **Machine Learning Pipeline (Ultralytics YOLO)**
   - Located in `/ml`
   - Written in pure Python
   - Modularized scripts for data merging, validation, and PyTorch training
   - Configured via `ml/config/paths.py` to ensure paths never conflict

## High-Level Flow
1. User uploads images via FE -> BE saves to `ml/data/review_queue/`.
2. User reviews detections in FE -> BE updates `ml/data/reviewed/`.
3. User triggers training -> BE spawns a background thread running `ml/scripts/active_learning_pipeline.py`.
4. ML Script merges labels, trains model, archives old weights, and outputs new `best.pt`.
5. BE detects new `best.pt` on the next prediction call (or via explicit reload) and swaps the loaded tensor in memory.
