"""Dataset and data loading utilities for Face Emotion Recognition.

Supports FER2013 dataset format (CSV with pixel values) and directory-based
image datasets with per-class folders.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


class FER2013Dataset(Dataset):
    """Dataset loader for FER2013 CSV format.

    FER2013 CSV columns: emotion, pixels, Usage
    - emotion: integer label (0-6)
    - pixels: space-separated pixel values (48x48)
    - Usage: Training, PublicTest, PrivateTest
    """

    def __init__(self, csv_path: str, split: str = "Training", transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Usage"] == split:
                    pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
                    self.images.append(pixels)
                    self.labels.append(int(row["emotion"]))

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # 48x48 uint8
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default: resize and normalize
            image = Image.fromarray(image)
            image = image.resize((config.IMG_SIZE, config.IMG_SIZE))
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.tensor(image).unsqueeze(0)

        return image, label


class ImageFolderDataset(Dataset):
    """Dataset loader for directory-based image datasets.

    Expected structure:
        data/train/
            angry/
            disgust/
            fear/
            happy/
            neutral/
            sad/
            surprise/
    """

    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {emotion: idx for idx, emotion in enumerate(config.EMOTIONS)}

        for emotion in config.EMOTIONS:
            emotion_dir = os.path.join(root_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            for img_name in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_name)
                if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[emotion])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = np.array(Image.open(img_path).convert("L"))  # Grayscale

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = Image.fromarray(image)
            image = image.resize((config.IMG_SIZE, config.IMG_SIZE))
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.tensor(image).unsqueeze(0)

        return image, label


def get_train_transforms():
    """Augmentation pipeline for training data."""
    return A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(0.1, 0.15), hole_width_range=(0.1, 0.15), p=0.2),  # Simulates occlusion
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Transform pipeline for validation/test data (no augmentation)."""
    return A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


def get_dataloaders(data_source: str = "folder", csv_path: str = None,
                    batch_size: int = None, num_workers: int = 4):
    """Create train and validation data loaders.

    Args:
        data_source: 'folder' for directory-based, 'csv' for FER2013 CSV
        csv_path: Path to FER2013 CSV file (required if data_source='csv')
        batch_size: Batch size (defaults to config.BATCH_SIZE)
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    if data_source == "csv":
        if csv_path is None:
            csv_path = os.path.join(config.DATA_DIR, "fer2013.csv")
        train_dataset = FER2013Dataset(csv_path, split="Training", transform=train_transform)
        val_dataset = FER2013Dataset(csv_path, split="PublicTest", transform=val_transform)
    else:
        train_dataset = ImageFolderDataset(config.TRAIN_DIR, transform=train_transform)
        val_dataset = ImageFolderDataset(config.TEST_DIR, transform=val_transform)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader
