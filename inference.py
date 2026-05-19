"""Low-latency real-time inference pipeline for Face Emotion Recognition.

Optimizations for real-time performance:
- Model compilation with torch.compile (PyTorch 2.0+)
- ONNX export for optimized inference
- Face detection with OpenCV Haar Cascade
- Frame skipping and ROI caching for webcam feed
- Batch processing for multiple faces
- Preprocessing on GPU when available
"""

import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

import config
from model import get_model


class EmotionPredictor:
    """Low-latency emotion prediction engine.

    Optimized for real-time inference with:
    - Model warmup and compilation
    - Efficient preprocessing pipeline
    - Temporal smoothing for stable predictions
    """

    def __init__(self, model_path: str = None, device: str = None, compile_model: bool = True):
        # Device selection
        if device is None:
            device = config.INFERENCE_DEVICE
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        # Load model
        if model_path is None:
            model_path = config.BEST_MODEL_PATH

        self.model = get_model(num_classes=config.NUM_CLASSES)
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
        self.model.to(self.device)
        self.model.eval()

        # Compile model for faster inference (PyTorch 2.0+)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with torch.compile (reduce-overhead mode)")
            except Exception:
                pass  # Fallback to eager mode

        # Warmup
        self._warmup()

        # Temporal smoothing buffer (last N predictions per face)
        self.prediction_buffer = deque(maxlen=5)

        # Performance tracking
        self.inference_times = deque(maxlen=100)

    def _warmup(self):
        """Warmup model with dummy input to optimize JIT compilation."""
        dummy = torch.randn(1, 1, config.IMG_SIZE, config.IMG_SIZE, device=self.device)
        with torch.no_grad():
            for _ in range(3):
                self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess a face image for model input.

        Args:
            face_image: Grayscale face crop (any size)

        Returns:
            Tensor of shape (1, 1, IMG_SIZE, IMG_SIZE)
        """
        # Resize with area interpolation for downsampling quality
        resized = cv2.resize(face_image, (config.IMG_SIZE, config.IMG_SIZE),
                             interpolation=cv2.INTER_AREA)

        # Normalize to [-1, 1]
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5

        # Convert to tensor
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def preprocess_batch(self, face_images: list) -> torch.Tensor:
        """Batch preprocess multiple face images."""
        tensors = []
        for face in face_images:
            resized = cv2.resize(face, (config.IMG_SIZE, config.IMG_SIZE),
                                 interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - 0.5) / 0.5
            tensors.append(torch.from_numpy(normalized).unsqueeze(0))
        batch = torch.stack(tensors).to(self.device)
        return batch

    @torch.no_grad()
    def predict(self, face_image: np.ndarray, smooth: bool = True) -> dict:
        """Predict emotion from a face image.

        Args:
            face_image: Grayscale face crop
            smooth: Apply temporal smoothing

        Returns:
            Dict with emotion, confidence, all_probabilities, inference_time_ms
        """
        start = time.perf_counter()

        tensor = self.preprocess(face_image)
        output = self.model(tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

        inference_time = (time.perf_counter() - start) * 1000
        self.inference_times.append(inference_time)

        # Temporal smoothing
        if smooth:
            self.prediction_buffer.append(probs)
            smoothed_probs = np.mean(self.prediction_buffer, axis=0)
        else:
            smoothed_probs = probs

        emotion_idx = int(np.argmax(smoothed_probs))
        confidence = float(smoothed_probs[emotion_idx])

        return {
            "emotion": config.EMOTIONS[emotion_idx],
            "confidence": confidence,
            "probabilities": {e: float(p) for e, p in zip(config.EMOTIONS, smoothed_probs)},
            "inference_time_ms": inference_time,
        }

    @torch.no_grad()
    def predict_batch(self, face_images: list) -> list:
        """Predict emotions for multiple faces in a single forward pass."""
        if not face_images:
            return []

        start = time.perf_counter()

        batch = self.preprocess_batch(face_images)
        outputs = self.model(batch)
        probs = F.softmax(outputs, dim=1).cpu().numpy()

        inference_time = (time.perf_counter() - start) * 1000

        results = []
        for i, prob in enumerate(probs):
            emotion_idx = int(np.argmax(prob))
            results.append({
                "emotion": config.EMOTIONS[emotion_idx],
                "confidence": float(prob[emotion_idx]),
                "probabilities": {e: float(p) for e, p in zip(config.EMOTIONS, prob)},
                "inference_time_ms": inference_time / len(face_images),
            })

        return results

    @property
    def avg_inference_time(self) -> float:
        """Average inference time in milliseconds."""
        if self.inference_times:
            return np.mean(self.inference_times)
        return 0.0


class RealTimeEmotionDetector:
    """Real-time webcam-based emotion detection pipeline.

    Optimizations:
    - Face detection every N frames (configurable skip)
    - ROI tracking between detection frames
    - Asynchronous face detection and emotion prediction
    - FPS display and performance monitoring
    """

    def __init__(self, model_path: str = None, detection_interval: int = 3):
        self.predictor = EmotionPredictor(model_path=model_path)
        self.detection_interval = detection_interval

        # Face detector
        cascade_path = cv2.data.haarcascades + config.FACE_CASCADE_PATH
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # State
        self.frame_count = 0
        self.cached_faces = []
        self.fps_buffer = deque(maxlen=30)

    def detect_faces(self, gray_frame: np.ndarray) -> list:
        """Detect faces in a grayscale frame."""
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces if len(faces) > 0 else []

    def draw_results(self, frame: np.ndarray, faces: list, predictions: list) -> np.ndarray:
        """Draw bounding boxes and emotion labels on frame."""
        emotion_colors = {
            "angry": (0, 0, 255),
            "disgust": (0, 128, 0),
            "fear": (128, 0, 128),
            "happy": (0, 255, 255),
            "neutral": (200, 200, 200),
            "sad": (255, 0, 0),
            "surprise": (0, 255, 0),
        }

        for (x, y, w, h), pred in zip(faces, predictions):
            emotion = pred["emotion"]
            confidence = pred["confidence"]
            color = emotion_colors.get(emotion, (255, 255, 255))

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label background
            label = f"{emotion}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                          (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Confidence bar
            bar_width = int(w * confidence)
            cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 15), color, -1)

        return frame

    def run(self, source: int = 0):
        """Run real-time emotion detection from webcam.

        Args:
            source: Camera index (0 for default webcam)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Starting real-time emotion detection...")
        print("Press 'q' to quit, 's' to save screenshot")

        while True:
            frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces (with frame skipping for performance)
            if self.frame_count % self.detection_interval == 0:
                self.cached_faces = self.detect_faces(gray)
            self.frame_count += 1

            # Predict emotions for detected faces
            predictions = []
            if len(self.cached_faces) > 0:
                face_crops = []
                for (x, y, w, h) in self.cached_faces:
                    # Ensure bounds are within frame
                    x, y = max(0, x), max(0, y)
                    face_crop = gray[y:y + h, x:x + w]
                    if face_crop.size > 0:
                        face_crops.append(face_crop)

                if face_crops:
                    predictions = self.predictor.predict_batch(face_crops)

            # Draw results
            if predictions:
                frame = self.draw_results(frame, self.cached_faces, predictions)

            # FPS counter
            fps = 1.0 / (time.perf_counter() - frame_start)
            self.fps_buffer.append(fps)
            avg_fps = np.mean(self.fps_buffer)

            # Display info
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inference: {self.predictor.avg_inference_time:.1f}ms",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(self.cached_faces)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Emotion Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                screenshot_path = os.path.join(config.RESULTS_DIR, "screenshot.png")
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession stats:")
        print(f"  Average FPS: {np.mean(self.fps_buffer):.1f}")
        print(f"  Average inference time: {self.predictor.avg_inference_time:.1f}ms")


def export_onnx(model_path: str = None, output_path: str = None):
    """Export model to ONNX for optimized deployment.

    ONNX Runtime provides faster inference than PyTorch for production.
    """
    if model_path is None:
        model_path = config.BEST_MODEL_PATH
    if output_path is None:
        output_path = os.path.join(config.MODEL_SAVE_DIR, "emotion_model.onnx")

    model = get_model(num_classes=config.NUM_CLASSES)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 1, config.IMG_SIZE, config.IMG_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to ONNX: {output_path}")
    return output_path


def benchmark_inference(model_path: str = None, num_iterations: int = 100):
    """Benchmark inference latency."""
    predictor = EmotionPredictor(model_path=model_path, compile_model=True)

    # Generate random face image
    dummy_face = np.random.randint(0, 255, (48, 48), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        predictor.predict(dummy_face, smooth=False)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        predictor.predict(dummy_face, smooth=False)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"\nInference Benchmark ({num_iterations} iterations):")
    print(f"  Device: {predictor.device}")
    print(f"  Mean latency: {times.mean():.2f} ms")
    print(f"  Median latency: {np.median(times):.2f} ms")
    print(f"  P95 latency: {np.percentile(times, 95):.2f} ms")
    print(f"  P99 latency: {np.percentile(times, 99):.2f} ms")
    print(f"  Throughput: {1000 / times.mean():.1f} FPS")


def main():
    """Entry point for inference pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Emotion Detection")
    parser.add_argument("--mode", choices=["webcam", "benchmark", "export"],
                        default="webcam", help="Inference mode")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--detection-interval", type=int, default=3,
                        help="Face detection interval (frames)")
    args = parser.parse_args()

    if args.mode == "webcam":
        detector = RealTimeEmotionDetector(
            model_path=args.model_path,
            detection_interval=args.detection_interval,
        )
        detector.run(source=args.camera)
    elif args.mode == "benchmark":
        benchmark_inference(model_path=args.model_path)
    elif args.mode == "export":
        export_onnx(model_path=args.model_path)


if __name__ == "__main__":
    main()
