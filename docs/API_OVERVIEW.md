# API Overview

The TrainFlowVision backend uses FastAPI. All routes are prefixed with `/api/v1`.

## Endpoints

### Project Management (`/api/v1/project`)
- `POST /upload`: Upload bulk images into the `review_queue`.
- `POST /init`: Upload a Label Studio ZIP to bootstrap the project and start initial training.
- `POST /refine`: Manually trigger the background active learning loop.
- `POST /annotate`: Save a reviewed annotation payload (bounding boxes + classes).
- `POST /skip`: Move an image from the queue to the `skipped` folder.
- `POST /reset`: Wipe or archive the current dataset and model runs.
- `GET /pending-images`: List filenames currently in the `review_queue`.
- `GET /classes`: Return the active taxonomy list.
- `POST /classes`: Add a new string to the ontology.

### Inference (`/api/v1/inference`)
- `GET /predict?image={filename}`: Run the currently loaded active model against an image in the queue and return bounding box JSON data.

### Batch Processing
- `POST /batch/queue`: Add an annotation to the temporary memory queue.
- `POST /batch/accept-all`: Dump the queue to disk and trigger training.
