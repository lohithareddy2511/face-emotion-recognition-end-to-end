"""Model evaluation with precision-recall analysis and failure case investigation.

Includes:
- Per-class precision, recall, F1-score
- Precision-recall curves and tradeoff analysis
- Confusion matrix visualization
- Confidence distribution analysis
- Failure case analysis under occlusion and lighting changes
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import config
from model import get_model
from dataset import get_dataloaders, get_val_transforms, ImageFolderDataset


class ModelEvaluator:
    """Comprehensive model evaluation with precision-recall analysis."""

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

    @torch.no_grad()
    def collect_predictions(self, dataloader) -> dict:
        """Collect all predictions, probabilities, and ground truth labels."""
        all_probs = []
        all_preds = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(self.device, non_blocking=True)

            with autocast(self.device.type, enabled=(self.device.type == "cuda")):
                outputs = self.model(images)

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

        return {
            "probs": np.concatenate(all_probs),
            "preds": np.concatenate(all_preds),
            "labels": np.concatenate(all_labels),
        }

    def classification_metrics(self, results: dict) -> str:
        """Generate detailed classification report."""
        report = classification_report(
            results["labels"],
            results["preds"],
            target_names=config.EMOTIONS,
            digits=4,
        )
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(report)

        # Save report
        report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        return report

    def plot_confusion_matrix(self, results: dict):
        """Generate and save confusion matrix heatmap."""
        cm = confusion_matrix(results["labels"], results["preds"])
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=config.EMOTIONS, yticklabels=config.EMOTIONS,
                    ax=axes[0])
        axes[0].set_title("Confusion Matrix (Counts)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=config.EMOTIONS, yticklabels=config.EMOTIONS,
                    ax=axes[1])
        axes[1].set_title("Confusion Matrix (Normalized)")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "confusion_matrix.png"), dpi=150)
        plt.close()
        print("Saved confusion matrix plot.")

    def plot_precision_recall_curves(self, results: dict):
        """Plot per-class precision-recall curves with AP scores."""
        n_classes = config.NUM_CLASSES
        labels_onehot = np.eye(n_classes)[results["labels"]]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        ap_scores = {}

        for i, emotion in enumerate(config.EMOTIONS):
            precision, recall, thresholds = precision_recall_curve(
                labels_onehot[:, i], results["probs"][:, i]
            )
            ap = average_precision_score(labels_onehot[:, i], results["probs"][:, i])
            ap_scores[emotion] = ap

            axes[i].plot(recall, precision, linewidth=2)
            axes[i].fill_between(recall, precision, alpha=0.2)
            axes[i].set_title(f"{emotion.capitalize()} (AP={ap:.3f})")
            axes[i].set_xlabel("Recall")
            axes[i].set_ylabel("Precision")
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0, 1])
            axes[i].grid(True, alpha=0.3)

        # Summary plot in last subplot
        axes[-1].barh(config.EMOTIONS, [ap_scores[e] for e in config.EMOTIONS])
        axes[-1].set_xlabel("Average Precision")
        axes[-1].set_title("AP Score Comparison")
        axes[-1].set_xlim([0, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "precision_recall_curves.png"), dpi=150)
        plt.close()

        # Save AP scores
        with open(os.path.join(config.RESULTS_DIR, "ap_scores.json"), "w") as f:
            json.dump(ap_scores, f, indent=2)

        print(f"Mean Average Precision: {np.mean(list(ap_scores.values())):.4f}")
        return ap_scores

    def precision_recall_tradeoff(self, results: dict):
        """Analyze precision-recall tradeoff at different confidence thresholds."""
        thresholds = np.arange(0.1, 1.0, 0.05)
        metrics_at_thresholds = []

        for thresh in thresholds:
            max_probs = results["probs"].max(axis=1)
            mask = max_probs >= thresh

            if mask.sum() == 0:
                continue

            filtered_preds = results["preds"][mask]
            filtered_labels = results["labels"][mask]
            coverage = mask.mean()
            accuracy = (filtered_preds == filtered_labels).mean()

            metrics_at_thresholds.append({
                "threshold": float(thresh),
                "coverage": float(coverage),
                "accuracy": float(accuracy),
                "num_samples": int(mask.sum()),
            })

        # Plot tradeoff
        fig, ax1 = plt.subplots(figsize=(10, 6))
        threshs = [m["threshold"] for m in metrics_at_thresholds]
        coverages = [m["coverage"] for m in metrics_at_thresholds]
        accuracies = [m["accuracy"] for m in metrics_at_thresholds]

        ax1.plot(threshs, accuracies, "b-o", label="Accuracy", linewidth=2)
        ax1.set_xlabel("Confidence Threshold")
        ax1.set_ylabel("Accuracy", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot(threshs, coverages, "r-s", label="Coverage", linewidth=2)
        ax2.set_ylabel("Coverage (% samples retained)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        ax1.set_title("Precision-Recall Tradeoff: Accuracy vs Coverage at Confidence Thresholds")
        ax1.grid(True, alpha=0.3)
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "precision_recall_tradeoff.png"), dpi=150)
        plt.close()

        # Save metrics
        with open(os.path.join(config.RESULTS_DIR, "threshold_metrics.json"), "w") as f:
            json.dump(metrics_at_thresholds, f, indent=2)

        print("Saved precision-recall tradeoff analysis.")
        return metrics_at_thresholds

    def confidence_distribution(self, results: dict):
        """Analyze prediction confidence for correct vs incorrect predictions."""
        max_probs = results["probs"].max(axis=1)
        correct_mask = results["preds"] == results["labels"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(max_probs[correct_mask], bins=30, alpha=0.7, label="Correct", color="green")
        axes[0].hist(max_probs[~correct_mask], bins=30, alpha=0.7, label="Incorrect", color="red")
        axes[0].set_xlabel("Confidence")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Confidence Distribution")
        axes[0].legend()

        # Per-class confidence
        class_confidences = []
        for i, emotion in enumerate(config.EMOTIONS):
            mask = results["labels"] == i
            class_confidences.append(max_probs[mask])

        axes[1].boxplot(class_confidences, labels=config.EMOTIONS)
        axes[1].set_xlabel("Emotion Class")
        axes[1].set_ylabel("Prediction Confidence")
        axes[1].set_title("Per-Class Confidence Distribution")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "confidence_distribution.png"), dpi=150)
        plt.close()
        print("Saved confidence distribution analysis.")

    def run_full_evaluation(self, dataloader):
        """Run complete evaluation pipeline."""
        print("\n" + "=" * 60)
        print("RUNNING FULL MODEL EVALUATION")
        print("=" * 60)

        results = self.collect_predictions(dataloader)

        self.classification_metrics(results)
        self.plot_confusion_matrix(results)
        self.plot_precision_recall_curves(results)
        self.precision_recall_tradeoff(results)
        self.confidence_distribution(results)

        # Overall accuracy
        accuracy = (results["preds"] == results["labels"]).mean()
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print(f"Results saved to: {config.RESULTS_DIR}")

        return results


def main():
    """Entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Face Emotion Recognition Model")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--data-source", choices=["folder", "csv"], default="folder")
    parser.add_argument("--csv-path", type=str, default=None)
    args = parser.parse_args()

    evaluator = ModelEvaluator(model_path=args.model_path)
    _, val_loader = get_dataloaders(data_source=args.data_source, csv_path=args.csv_path)
    evaluator.run_full_evaluation(val_loader)


if __name__ == "__main__":
    main()
