# Local Development Setup

Follow these steps to run TrainFlowVision locally.

## Prerequisites
- Python 3.11+
- Node.js 20+
- Optional: CUDA-compatible GPU

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd TrainFlowVision
   ```

2. **Run the Initialization Script**
   TrainFlowVision uses a unified script that handles both Python and Node.js environments.
   ```bash
   python run_dev.py
   ```
   This script will:
   - Validate directory structure
   - Install missing Python packages from `requirements.txt`
   - Install `node_modules` in the `FE` directory if missing
   - Start the FastAPI backend on `http://localhost:8000`
   - Start the Angular frontend on `http://localhost:4200`

## Stopping the Server
Press `Ctrl+C` in the terminal where `run_dev.py` is running. The script will cleanly shutdown both the backend and frontend processes.
