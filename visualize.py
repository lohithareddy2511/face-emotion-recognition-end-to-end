"""Visualization utilities for training curves and results."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

import config


def plot_training_curves(history_path: str = None):
    """Plot training and validation loss/accuracy curves."""
    if history_path is None:
        history_path = os.path.join(config.RESULTS_DIR, "training_history.json")

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(epochs, history["lr"], "g-", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved training curves plot.")


if __name__ == "__main__":
    plot_training_curves()
