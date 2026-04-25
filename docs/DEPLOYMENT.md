# Deployment Guide for TrainFlowVision

This guide provides instructions for deploying TrainFlowVision in a production environment.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA drivers installed (optional but highly recommended for ML inference and training)

## Deploying with Docker Compose

1. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and adjust the variables as needed.
   ```bash
   cp .env.example .env
   ```

2. **Build and Run:**
   Run the following command to build the Docker images and start the containers in detached mode:
   ```bash
   docker-compose up --build -d
   ```

3. **Accessing the Application:**
   - Frontend: `http://localhost:4200` (or the `FE_PORT` specified in your `.env`)
   - Backend API: `http://localhost:8000` (or the `BE_PORT` specified in your `.env`)

## Local Setup without Docker

To run locally without Docker, refer to the `DEV_SETUP.md` guide. You can also run the provided `python run_dev.py` script.

## Notes

- **Volumes**: The `docker-compose.yml` mounts the `./ml` directory to ensure model data and datasets persist across container restarts.
- **CUDA**: Make sure to pass the appropriate `--gpus all` flags or adjust the docker-compose to use the `nvidia` runtime if you want hardware acceleration inside the container.
