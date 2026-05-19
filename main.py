"""Face Emotion Recognition System - Main Entry Point.

Usage:
    python main.py train --data-source csv --csv-path data/fer2013.csv
    python main.py evaluate --model-path checkpoints/best_model.pth
    python main.py analyze --model-path checkpoints/best_model.pth
    python main.py infer --mode webcam
    python main.py infer --mode benchmark
    python main.py export
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Face Emotion Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train model:       python main.py train --data-source csv --csv-path data/fer2013.csv
  Evaluate model:    python main.py evaluate
  Failure analysis:  python main.py analyze
  Real-time webcam:  python main.py infer --mode webcam
  Benchmark:         python main.py infer --mode benchmark
  Export ONNX:       python main.py export
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-source", choices=["folder", "csv"], default="folder")
    train_parser.add_argument("--csv-path", type=str, default=None)
    train_parser.add_argument("--resume", type=str, default=None)

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--model-path", type=str, default=None)
    eval_parser.add_argument("--data-source", choices=["folder", "csv"], default="folder")
    eval_parser.add_argument("--csv-path", type=str, default=None)

    # Failure Analysis
    analyze_parser = subparsers.add_parser("analyze", help="Run failure case analysis")
    analyze_parser.add_argument("--model-path", type=str, default=None)
    analyze_parser.add_argument("--data-source", choices=["folder", "csv"], default="folder")
    analyze_parser.add_argument("--csv-path", type=str, default=None)

    # Inference
    infer_parser = subparsers.add_parser("infer", help="Run inference pipeline")
    infer_parser.add_argument("--mode", choices=["webcam", "benchmark"], default="webcam")
    infer_parser.add_argument("--model-path", type=str, default=None)
    infer_parser.add_argument("--camera", type=int, default=0)
    infer_parser.add_argument("--detection-interval", type=int, default=3)

    # Export
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("--model-path", type=str, default=None)
    export_parser.add_argument("--output-path", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        from train import Trainer
        trainer = Trainer(
            data_source=args.data_source,
            csv_path=args.csv_path,
            resume_path=args.resume,
        )
        trainer.train()

    elif args.command == "evaluate":
        from evaluate import ModelEvaluator
        from dataset import get_dataloaders
        evaluator = ModelEvaluator(model_path=args.model_path)
        _, val_loader = get_dataloaders(data_source=args.data_source, csv_path=args.csv_path)
        evaluator.run_full_evaluation(val_loader)

    elif args.command == "analyze":
        from failure_analysis import FailureCaseAnalyzer
        from dataset import get_dataloaders
        analyzer = FailureCaseAnalyzer(model_path=args.model_path)
        _, val_loader = get_dataloaders(data_source=args.data_source, csv_path=args.csv_path)
        analyzer.run_full_analysis(val_loader)

    elif args.command == "infer":
        from inference import RealTimeEmotionDetector, benchmark_inference
        if args.mode == "webcam":
            detector = RealTimeEmotionDetector(
                model_path=args.model_path,
                detection_interval=args.detection_interval,
            )
            detector.run(source=args.camera)
        else:
            benchmark_inference(model_path=args.model_path)

    elif args.command == "export":
        from inference import export_onnx
        export_onnx(model_path=args.model_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
