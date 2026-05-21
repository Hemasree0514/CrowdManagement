"""
backend/models/crowd_detector.py
YOLOv8-based people counting and density analysis.

What it does:
  1. Receives a BGR frame from CameraManager
  2. Runs YOLOv8 inference (class 0 = person)
  3. Returns count + bounding boxes for the frame

Install:
    pip install ultralytics
    # Model downloads automatically on first run (~6 MB for yolov8n)
"""
import cv2
import numpy as np
from dataclasses import dataclass
from backend.utils.logger import log

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    log.warning("ultralytics_not_installed", hint="pip install ultralytics")


@dataclass
class DetectionResult:
    count:        int
    boxes:        list    # list of [x1, y1, x2, y2, confidence]
    annotated_frame: np.ndarray | None = None


class CrowdDetector:
    """
    Wraps YOLOv8 for people detection.

    model_name options (speed vs accuracy tradeoff):
        yolov8n.pt  — nano,  fastest,  less accurate  (edge devices)
        yolov8s.pt  — small, fast,     good accuracy   (recommended)
        yolov8m.pt  — medium, slower,  higher accuracy
        yolov8l.pt  — large, slowest,  best accuracy   (GPU required)
    """

    def __init__(self, model_name: str = "yolov8s.pt", confidence: float = 0.4):
        self.confidence  = confidence
        self.model       = None
        if _YOLO_AVAILABLE:
            self.model = YOLO(model_name)
            log.info("yolo_model_loaded", model=model_name)
        else:
            log.warning("crowd_detector_running_in_mock_mode")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a single frame.
        Returns DetectionResult with count + bounding boxes.
        """
        if frame is None:
            return DetectionResult(count=0, boxes=[])

        if self.model is None:
            # Mock mode — return dummy data for development
            return self._mock_detect(frame)

        results = self.model(
            frame,
            classes=[0],          # class 0 = person in COCO dataset
            conf=self.confidence,
            verbose=False,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append([x1, y1, x2, y2, conf])

        annotated = results[0].plot() if results else frame

        return DetectionResult(
            count=len(boxes),
            boxes=boxes,
            annotated_frame=annotated,
        )

    def _mock_detect(self, frame: np.ndarray) -> DetectionResult:
        """Returns dummy detection — used when YOLO is not installed."""
        h, w = frame.shape[:2]
        count = np.random.randint(10, 80)
        boxes = [
            [
                np.random.randint(0, w - 50),
                np.random.randint(0, h - 100),
                np.random.randint(50, w),
                np.random.randint(100, h),
                round(np.random.uniform(0.4, 0.99), 2),
            ]
            for _ in range(count)
        ]
        return DetectionResult(count=count, boxes=boxes, annotated_frame=frame)

    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw bounding boxes and count overlay on frame.
        Returns annotated frame for display / streaming.
        """
        if result.annotated_frame is not None:
            return result.annotated_frame

        out = frame.copy()
        for x1, y1, x2, y2, conf in result.boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 100), 2)
            cv2.putText(out, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 100), 1)

        # Count overlay
        cv2.putText(out, f"Count: {result.count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return out
