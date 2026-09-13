# eye-fatigue-estimator

## Overview
A CPU-friendly PyTorch machine learning pipeline for predicting a continuous eye fatigue score (0-100) from face images.
The project uses Transfer Learning with MobileNetV2 and relies on MediaPipe for automated label generation and face cropping during data preparation.

##  Key Features
- **Neural Network Regression**: Outputs a continuous fatigue score (0 = Fully Alert, 100 = Very Fatigued) using a custom MobileNetV2 architecture.
- **CPU Optimized**: Designed for fast, real-time inference on standard laptops without requiring a GPU.
- **Auto-Labeling Pipeline**: Extracts Eye Aspect Ratio (EAR) via MediaPipe Face Mesh to generate training labels automatically from raw images.
- **Robust Evaluation**: Uses sequential dataset splitting to prevent data leakage and evaluates purely on a hold-out validation set.
- **Live Webcam Dashboard**: Real-time evaluation using OpenCV and PyTorch. Features include:
  - **Color-Coded Status**: Green (Alert), Yellow (Tired), Red (Fatigued).
  - **Smoothed EAR Tracking**: Filters out normal blinks so your fatigue score doesn't spike artificially.
  - **Drowsiness Alarm**: A background audio alarm triggers if your eyes remain closed for more than 2 seconds.
  - **Full Screen HUD**: Provides an immersive heads-up display of your stats.

##  Project Structure
```
project_root/
│
├── run.py                #  Main pipeline orchestrator (Run this!)
├── config.py             # Hyperparameters, paths, and EAR mapping
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
│
├── data/
│   ├── raw/              # Place raw face/eye images here before running data_prep.py
│   └── processed/        # Cropped and resized images (generated automatically)
│
├── models/               # Directory for saving trained model checkpoints
│
└── src/
    ├── data_prep.py      # Computes EAR, crops faces, and auto-labels dataset
    ├── dataset.py        # PyTorch Dataset and Transforms
    ├── model.py          # MobileNetV2 Regression Model (Sigmoid + Scale)
    ├── train.py          # Training loop with Sequential Split & MSE Loss
    ├── evaluate.py       # Metrics and scatter plot visualization
    └── inference.py      # Real-time webcam script powered by PyTorch
```

## ️ Setup Instructions

### 1. Create a Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

##  How to Run

The easiest way to use the project is through the `run.py` orchestrator script. It handles everything from data preparation to live inference.

### Option A: Run the Full Pipeline
If you have just added new raw images to `data/raw/` and want to process them, train the model, evaluate it, and jump into the webcam dashboard, simply run:
```bash
python run.py --mode all
```

### Option B: Live Webcam Inference Only (Recommended for Testing)
If you already have a trained model (`models/best_model.pth`) and just want to test the real-time eye fatigue estimator:
```bash
python run.py --mode live-only
```
*Press `q` on your keyboard to exit the webcam window.*

### Option C: Train and Evaluate Only
If you want to prepare data, train the PyTorch model, and evaluate its performance (saving an evaluation plot) without opening the webcam:
```bash
python run.py --mode train-only
```

## ️ Customization
You can easily customize the behavior of the network and the pipeline by modifying `config.py`:
- **Hyperparameters:** Learning Rate, Epochs, Batch Size
- **Image Size:** Default is `224x224` (MobileNetV2 standard)
- **EAR Mapping:** Adjust `EAR_MAX` (alert) and `EAR_MIN` (fatigued) if the system is too sensitive or not sensitive enough for your specific eye shape.

## Getting Started
Please refer to the source files for specific installation and usage instructions. Ensure that your local environment meets the standard requirements for the associated technologies.

## Project Structure
This project is organized into standard directories. Key configuration files and primary source code are located in the root directory.
