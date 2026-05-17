# TrainFlowVision - Active Learning Platform

An intelligent object detection system using YOLOv8 and active learning.

## 🚀 Quick Start

### Windows
Double-click `start_dev.bat` or run:
```bash
start_dev.bat
```

### Linux/Mac
```bash
chmod +x start_dev.sh
./start_dev.sh
```

### Manual (Recommended for Development)
See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## 📖 Documentation

- [Quick Start Guide](QUICKSTART.md) - How to run the project
- [Walkthrough](C:\Users\admin\.gemini\antigravity\brain\9ceec566-0862-4806-bc8f-3b09a8c7f6e9\walkthrough.md) - Development history and features

## 🌐 Access Points

- **Frontend**: http://localhost:4200
- **Backend API**: http://localhost:8000  
- **API Docs**: http://localhost:8000/docs

## 🎯 Features

- Drag & drop ZIP upload for Label Studio exports
- Real-time training with GPU acceleration (CUDA)
- Live training logs streaming to UI
- Resizable layout with splitters
- System dashboard with run history
- Model rollback capability
- Fresh start / append mode for datasets

## 🛠️ Tech Stack

- **Backend**: FastAPI, PyTorch, YOLOv8, Ultralytics
- **Frontend**: Angular 17 (Standalone), TypeScript, SCSS
- **ML**: Active Learning, Object Detection

## 📝 Requirements

- Python 3.11
- Node.js 18+
- NVIDIA GPU with CUDA support (optional, falls back to CPU)
