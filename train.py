"""Training pipeline for Face Emotion Recognition model.

Includes:
- Training loop with mixed precision support
- Learning rate scheduling (cosine annealing)
- Early stopping
- Training metrics logging
- Model checkpointing
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

import config
from model import get_model
from dataset import get_dataloaders


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class Trainer:
    """Training orchestrator for the emotion recognition model."""

    def __init__(self, data_source: str = "folder", csv_path: str = None,
                 resume_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model
        self.model = get_model(
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT_RATE
        ).to(self.device)

        self.start_epoch = 1
        if resume_path and os.path.exists(resume_path):
            self.model.load_state_dict(
                torch.load(resume_path, map_location=self.device, weights_only=True)
            )
            # Extract epoch number from filename (e.g. epoch_25.pth -> 25)
            basename = os.path.basename(resume_path)
            if basename.startswith("epoch_"):
                self.start_epoch = int(basename.split("_")[1].split(".")[0]) + 1
            print(f"Resumed from checkpoint: {resume_path}, starting at epoch {self.start_epoch}")

        # Data
        self.train_loader, self.val_loader = get_dataloaders(
            data_source=data_source, csv_path=csv_path
        )

        # Loss with class weights to handle imbalance
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )

        # Mixed precision (only on CUDA)
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

        # Training history
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [],
        }

        # Create checkpoint directory
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    def train_epoch(self) -> tuple:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> tuple:
        """Run validation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f"epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        if is_best:
            torch.save(self.model.state_dict(), config.BEST_MODEL_PATH)

    def train(self):
        """Full training loop."""
        print(f"Starting training from epoch {self.start_epoch} to {config.NUM_EPOCHS}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print("-" * 60)

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(self.start_epoch, config.NUM_EPOCHS + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Save best model
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            self.save_checkpoint(epoch, is_best=is_best)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )

            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best validation accuracy: {best_val_acc:.4f}")

        # Save training history
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history


def main():
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Face Emotion Recognition Model")
    parser.add_argument("--data-source", choices=["folder", "csv"], default="folder",
                        help="Data source type")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to FER2013 CSV file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    trainer = Trainer(
        data_source=args.data_source,
        csv_path=args.csv_path,
        resume_path=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
