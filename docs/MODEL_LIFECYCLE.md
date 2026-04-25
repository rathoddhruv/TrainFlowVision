# Model Lifecycle

TrainFlowVision is an Active Learning platform. The neural engine evolves via a continuous feedback loop driven by human annotations.

## 1. Bootstrapping
When the app starts fresh, it falls back to a base model (`ml/models/base/yolov8n.pt`).
Alternatively, you can upload a ZIP file containing a Label Studio dataset to initialize the first epoch of custom training.

## 2. The Active Learning Loop
1. **Upload**: Images are uploaded and stored in `ml/data/review_queue/`.
2. **Review**: The backend (`current/best.pt`) predicts boxes on these images dynamically. A human reviewer accepts, modifies, or rejects them.
3. **Staging**: Validated annotations move to `ml/data/reviewed/`.
4. **Merge**: The `boost_merge_labels.py` script combines the newly reviewed data with the existing `training/` dataset.
5. **Retrain**: `active_learning_pipeline.py` fine-tunes the `current/best.pt` model on the expanded dataset.
6. **Archive**: The previous model is saved to `ml/models/history/` for rollback capability.
7. **Deploy**: The newly trained `best.pt` automatically replaces the file in `ml/models/current/`. The API instantly reloads the new weights into memory.
