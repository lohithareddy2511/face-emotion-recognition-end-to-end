"""Configuration for Face Emotion Recognition System."""

import os

# Dataset settings
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "archive", "train")
TEST_DIR = os.path.join(DATA_DIR, "archive", "test")

# Emotion classes (FER2013 standard)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# Image settings
IMG_SIZE = 227  # AlexNet input size
NUM_CHANNELS = 1  # Grayscale

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

# Model settings
DROPOUT_RATE = 0.5
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

# Inference settings
CONFIDENCE_THRESHOLD = 0.3
INFERENCE_DEVICE = "cuda"  # Will fallback to cpu if unavailable
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Evaluation settings
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
