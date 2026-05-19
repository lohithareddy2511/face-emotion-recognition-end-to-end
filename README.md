# Face Emotion Recognition System

A CNN-based (AlexNet) real-time emotion classification system that detects and classifies facial emotions from images and webcam feeds.

## Features

- **AlexNet CNN Model**: Modified AlexNet architecture adapted for grayscale facial emotion classification (7 classes)
- **Real-time Inference**: Low-latency pipeline with face detection, temporal smoothing, and batch processing
- **Comprehensive Evaluation**: Precision-recall curves, confusion matrices, and confidence analysis
- **Robustness Analysis**: Failure case analysis under occlusion and lighting variations
- **ONNX Export**: Model export for optimized production deployment

## Architecture

```
Input (1x227x227 grayscale)
    │
    ├── Conv1 (96 filters, 11x11, stride 4) + BN + ReLU + MaxPool
    ├── Conv2 (256 filters, 5x5) + BN + ReLU + MaxPool
    ├── Conv3 (384 filters, 3x3) + BN + ReLU
    ├── Conv4 (384 filters, 3x3) + BN + ReLU
    ├── Conv5 (256 filters, 3x3) + BN + ReLU + MaxPool
    │
    ├── AdaptiveAvgPool → 256x6x6
    │
    ├── FC1 (9216 → 4096) + Dropout(0.5) + ReLU
    ├── FC2 (4096 → 4096) + Dropout(0.5) + ReLU
    └── FC3 (4096 → 7) → Emotion Prediction
```

## Emotion Classes

| Index | Emotion  |
|-------|----------|
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Neutral  |
| 5     | Sad      |
| 6     | Surprise |

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This system supports two data formats:

### Option 1: FER2013 CSV (Recommended)
Download `fer2013.csv` from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it in `data/`:
```
data/
└── fer2013.csv
```

### Option 2: Image Folder Structure
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── ...
```

## Usage

### Training
```bash
# Train with FER2013 CSV
python main.py train --data-source csv --csv-path data/fer2013.csv

# Train with image folders
python main.py train --data-source folder

# Resume training from checkpoint
python main.py train --data-source csv --csv-path data/fer2013.csv --resume checkpoints/epoch_10.pth
```

### Evaluation
```bash
# Full evaluation with precision-recall analysis
python main.py evaluate --data-source csv --csv-path data/fer2013.csv

# Evaluate specific checkpoint
python main.py evaluate --model-path checkpoints/epoch_30.pth
```

### Failure Analysis
```bash
# Analyze robustness to occlusion and lighting
python main.py analyze --data-source csv --csv-path data/fer2013.csv
```

### Real-time Inference
```bash
# Webcam-based emotion detection
python main.py infer --mode webcam

# Benchmark inference latency
python main.py infer --mode benchmark
```

### Export Model
```bash
# Export to ONNX for production deployment
python main.py export
```

## Project Structure

```
├── config.py              # Configuration and hyperparameters
├── model.py               # AlexNet architecture definition
├── dataset.py             # Data loading and augmentation
├── train.py               # Training pipeline with mixed precision
├── evaluate.py            # Evaluation with precision-recall analysis
├── failure_analysis.py    # Occlusion and lighting robustness analysis
├── inference.py           # Real-time inference pipeline
├── visualize.py           # Training curve visualization
├── main.py                # Unified entry point
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Key Design Decisions

### Training Optimizations
- **Mixed Precision (AMP)**: 2x faster training on GPU with minimal accuracy loss
- **Cosine Annealing LR**: Smooth learning rate decay for better convergence
- **Label Smoothing (0.1)**: Reduces overconfidence, improves generalization
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Prevents overfitting with patience=7

### Data Augmentation
- Horizontal flip, rotation (±15°)
- Random brightness/contrast
- Gaussian noise (simulates low-light sensor noise)
- Coarse dropout (simulates occlusion during training)

### Inference Optimizations
- **torch.compile**: JIT compilation for reduced overhead
- **Frame skipping**: Face detection every N frames, reuses cached ROIs
- **Batch processing**: Multiple faces processed in single forward pass
- **Temporal smoothing**: Rolling average over last 5 predictions for stability
- **ONNX export**: For deployment with ONNX Runtime

## Results (Generated After Training)

After running evaluation, find results in `results/`:
- `classification_report.txt` - Per-class precision, recall, F1
- `confusion_matrix.png` - Confusion matrix visualization
- `precision_recall_curves.png` - Per-class PR curves with AP scores
- `precision_recall_tradeoff.png` - Accuracy vs coverage at different thresholds
- `confidence_distribution.png` - Confidence analysis for correct/incorrect predictions
- `occlusion_analysis.png` - Accuracy degradation under occlusion
- `lighting_analysis.png` - Robustness to brightness/contrast/noise
- `failure_patterns.png` - Most common misclassification pairs
