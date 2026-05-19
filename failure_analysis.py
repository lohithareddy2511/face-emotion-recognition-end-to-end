"""Failure case analysis under occlusion and lighting changes.

Systematically evaluates model robustness by:
1. Simulating occlusion (partial face coverage)
2. Simulating lighting changes (brightness, contrast, shadows)
3. Identifying and visualizing failure patterns
4. Generating actionable insights for model improvement
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import config
from model import get_model
from dataset import get_dataloaders, FER2013Dataset, ImageFolderDataset


class FailureCaseAnalyzer:
    """Analyze model failures under challenging conditions."""

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(num_classes=config.NUM_CLASSES)

        if model_path is None:
            model_path = config.BEST_MODEL_PATH

        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
        self.model.to(self.device)
        self.model.eval()

        os.makedirs(config.RESULTS_DIR, exist_ok=True)

    def _get_raw_images_and_labels(self, dataloader) -> tuple:
        """Extract raw images and labels from dataloader for re-augmentation."""
        all_images = []
        all_labels = []
        for images, labels in dataloader:
            all_images.append(images)
            all_labels.append(labels)
        return torch.cat(all_images), torch.cat(all_labels)

    @torch.no_grad()
    def _evaluate_batch(self, images: torch.Tensor) -> tuple:
        """Get predictions and confidence for a batch of images."""
        images = images.to(self.device)
        with autocast(self.device.type, enabled=(self.device.type == "cuda")):
            outputs = self.model(images)
        probs = F.softmax(outputs, dim=1)
        confidence, preds = probs.max(dim=1)
        return preds.cpu(), confidence.cpu(), probs.cpu()

    def analyze_occlusion(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """Test model robustness to various levels of occlusion.

        Simulates:
        - Lower face occlusion (mask wearing)
        - Upper face occlusion (sunglasses/hat)
        - Random rectangular occlusion
        - Varying occlusion percentages
        """
        results = {}
        occlusion_types = {
            "lower_face": self._apply_lower_occlusion,
            "upper_face": self._apply_upper_occlusion,
            "random_rect": self._apply_random_occlusion,
        }
        occlusion_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

        for occ_type, occ_fn in occlusion_types.items():
            type_results = []
            for level in occlusion_levels:
                occluded_images = occ_fn(images.clone(), level)
                preds, confidence, _ = self._evaluate_batch(occluded_images)
                accuracy = (preds == labels).float().mean().item()
                avg_confidence = confidence.mean().item()
                type_results.append({
                    "level": level,
                    "accuracy": accuracy,
                    "avg_confidence": avg_confidence,
                })
            results[occ_type] = type_results

        self._plot_occlusion_results(results)
        return results

    def _apply_lower_occlusion(self, images: torch.Tensor, level: float) -> torch.Tensor:
        """Apply occlusion to lower portion of face (simulates mask)."""
        h = images.shape[2]
        occ_height = int(h * level)
        images[:, :, h - occ_height:, :] = 0
        return images

    def _apply_upper_occlusion(self, images: torch.Tensor, level: float) -> torch.Tensor:
        """Apply occlusion to upper portion of face (simulates hat/sunglasses)."""
        h = images.shape[2]
        occ_height = int(h * level)
        images[:, :, :occ_height, :] = 0
        return images

    def _apply_random_occlusion(self, images: torch.Tensor, level: float) -> torch.Tensor:
        """Apply random rectangular occlusion."""
        b, c, h, w = images.shape
        occ_h = int(h * level)
        occ_w = int(w * level)
        for i in range(b):
            y = np.random.randint(0, h - occ_h)
            x = np.random.randint(0, w - occ_w)
            images[i, :, y:y + occ_h, x:x + occ_w] = 0
        return images

    def _plot_occlusion_results(self, results: dict):
        """Visualize occlusion analysis results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for occ_type, type_results in results.items():
            levels = [r["level"] for r in type_results]
            accuracies = [r["accuracy"] for r in type_results]
            confidences = [r["avg_confidence"] for r in type_results]

            axes[0].plot(levels, accuracies, "-o", label=occ_type, linewidth=2)
            axes[1].plot(levels, confidences, "-o", label=occ_type, linewidth=2)

        axes[0].set_xlabel("Occlusion Level (% of face covered)")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Model Accuracy vs Occlusion Level")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Occlusion Level (% of face covered)")
        axes[1].set_ylabel("Average Confidence")
        axes[1].set_title("Prediction Confidence vs Occlusion Level")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "occlusion_analysis.png"), dpi=150)
        plt.close()
        print("Saved occlusion analysis plot.")

    def analyze_lighting(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """Test model robustness to lighting variations.

        Simulates:
        - Brightness changes (dark and bright conditions)
        - Contrast reduction
        - Directional shadows
        """
        results = {}

        # Brightness variations
        brightness_factors = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
        brightness_results = []
        for factor in brightness_factors:
            modified = (images.clone() * factor).clamp(-1, 1)
            preds, confidence, _ = self._evaluate_batch(modified)
            accuracy = (preds == labels).float().mean().item()
            brightness_results.append({
                "factor": factor,
                "accuracy": accuracy,
                "avg_confidence": confidence.mean().item(),
            })
        results["brightness"] = brightness_results

        # Contrast variations
        contrast_factors = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
        contrast_results = []
        for factor in contrast_factors:
            mean = images.mean()
            modified = ((images.clone() - mean) * factor + mean).clamp(-1, 1)
            preds, confidence, _ = self._evaluate_batch(modified)
            accuracy = (preds == labels).float().mean().item()
            contrast_results.append({
                "factor": factor,
                "accuracy": accuracy,
                "avg_confidence": confidence.mean().item(),
            })
        results["contrast"] = contrast_results

        # Gaussian noise (simulates sensor noise in low light)
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
        noise_results = []
        for noise_std in noise_levels:
            noise = torch.randn_like(images) * noise_std
            modified = (images.clone() + noise).clamp(-1, 1)
            preds, confidence, _ = self._evaluate_batch(modified)
            accuracy = (preds == labels).float().mean().item()
            noise_results.append({
                "noise_std": noise_std,
                "accuracy": accuracy,
                "avg_confidence": confidence.mean().item(),
            })
        results["noise"] = noise_results

        self._plot_lighting_results(results)
        return results

    def _plot_lighting_results(self, results: dict):
        """Visualize lighting robustness analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Brightness
        factors = [r["factor"] for r in results["brightness"]]
        accs = [r["accuracy"] for r in results["brightness"]]
        axes[0].plot(factors, accs, "b-o", linewidth=2)
        axes[0].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Brightness Factor")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Robustness to Brightness Changes")
        axes[0].grid(True, alpha=0.3)

        # Contrast
        factors = [r["factor"] for r in results["contrast"]]
        accs = [r["accuracy"] for r in results["contrast"]]
        axes[1].plot(factors, accs, "g-o", linewidth=2)
        axes[1].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Contrast Factor")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Robustness to Contrast Changes")
        axes[1].grid(True, alpha=0.3)

        # Noise
        noise_stds = [r["noise_std"] for r in results["noise"]]
        accs = [r["accuracy"] for r in results["noise"]]
        axes[2].plot(noise_stds, accs, "r-o", linewidth=2)
        axes[2].set_xlabel("Noise Std Dev")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("Robustness to Sensor Noise (Low Light)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "lighting_analysis.png"), dpi=150)
        plt.close()
        print("Saved lighting analysis plot.")

    def identify_failure_patterns(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """Identify systematic failure patterns in predictions."""
        preds, confidence, probs = self._evaluate_batch(images)

        failures = preds != labels
        failure_indices = torch.where(failures)[0]

        # Analyze misclassification patterns
        failure_analysis = {
            "total_failures": int(failures.sum()),
            "failure_rate": float(failures.float().mean()),
            "avg_confidence_failures": float(confidence[failures].mean()) if failures.any() else 0,
            "avg_confidence_correct": float(confidence[~failures].mean()),
            "misclassification_pairs": {},
        }

        # Count misclassification pairs
        for idx in failure_indices:
            true_label = config.EMOTIONS[labels[idx]]
            pred_label = config.EMOTIONS[preds[idx]]
            pair = f"{true_label} -> {pred_label}"
            failure_analysis["misclassification_pairs"][pair] = (
                failure_analysis["misclassification_pairs"].get(pair, 0) + 1
            )

        # Sort by frequency
        failure_analysis["misclassification_pairs"] = dict(
            sorted(
                failure_analysis["misclassification_pairs"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # Visualize top failure patterns
        self._plot_failure_patterns(failure_analysis)

        # Save analysis
        import json
        with open(os.path.join(config.RESULTS_DIR, "failure_analysis.json"), "w") as f:
            json.dump(failure_analysis, f, indent=2)

        return failure_analysis

    def _plot_failure_patterns(self, analysis: dict):
        """Visualize most common misclassification patterns."""
        pairs = analysis["misclassification_pairs"]
        if not pairs:
            return

        top_pairs = list(pairs.items())[:10]
        labels_plot = [p[0] for p in top_pairs]
        counts = [p[1] for p in top_pairs]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(labels_plot, counts, color="coral")
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Misclassification Patterns (True -> Predicted)")
        ax.invert_yaxis()

        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center")

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "failure_patterns.png"), dpi=150)
        plt.close()
        print("Saved failure patterns plot.")

    def run_full_analysis(self, dataloader):
        """Run complete failure case analysis."""
        print("\n" + "=" * 60)
        print("FAILURE CASE ANALYSIS")
        print("=" * 60)

        # Collect a subset of data for analysis
        images, labels = self._get_raw_images_and_labels(dataloader)

        # Limit to manageable size for analysis
        max_samples = min(2000, len(labels))
        indices = torch.randperm(len(labels))[:max_samples]
        images = images[indices]
        labels = labels[indices]

        print(f"\nAnalyzing {max_samples} samples...")

        # Failure pattern analysis
        print("\n1. Identifying failure patterns...")
        failure_results = self.identify_failure_patterns(images, labels)
        print(f"   Failure rate: {failure_results['failure_rate']:.4f}")
        print(f"   Top misclassification: {list(failure_results['misclassification_pairs'].items())[:3]}")

        # Occlusion analysis
        print("\n2. Analyzing occlusion robustness...")
        occlusion_results = self.analyze_occlusion(images, labels)

        # Lighting analysis
        print("\n3. Analyzing lighting robustness...")
        lighting_results = self.analyze_lighting(images, labels)

        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Baseline accuracy: {1 - failure_results['failure_rate']:.4f}")
        print(f"Accuracy with 30% lower occlusion: {occlusion_results['lower_face'][2]['accuracy']:.4f}")
        print(f"Accuracy with 30% upper occlusion: {occlusion_results['upper_face'][2]['accuracy']:.4f}")
        print(f"Accuracy at 0.5x brightness: {lighting_results['brightness'][1]['accuracy']:.4f}")
        print(f"Accuracy at 0.1 noise std: {lighting_results['noise'][2]['accuracy']:.4f}")
        print(f"\nAll results saved to: {config.RESULTS_DIR}")

        return {
            "failures": failure_results,
            "occlusion": occlusion_results,
            "lighting": lighting_results,
        }


def main():
    """Entry point for failure analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze model failure cases")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-source", choices=["folder", "csv"], default="folder")
    parser.add_argument("--csv-path", type=str, default=None)
    args = parser.parse_args()

    analyzer = FailureCaseAnalyzer(model_path=args.model_path)
    _, val_loader = get_dataloaders(data_source=args.data_source, csv_path=args.csv_path)
    analyzer.run_full_analysis(val_loader)


if __name__ == "__main__":
    main()
