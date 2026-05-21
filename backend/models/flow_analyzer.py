"""
backend/models/flow_analyzer.py
Crowd direction tracking and bottleneck detection using optical flow.

What it does:
  - Uses Lucas-Kanade optical flow on consecutive frames
  - Computes average crowd movement vector (direction + speed)
  - Detects bottlenecks: high density + low movement speed
  - Detects panic signatures: sudden reverse / chaotic flow
"""
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum


class FlowAnomaly(str, Enum):
    NORMAL      = "normal"
    BOTTLENECK  = "bottleneck"    # dense + stuck
    SURGE       = "surge"         # rapid inflow
    PANIC       = "panic"         # reverse / chaotic flow
    DISPERSAL   = "dispersal"     # rapid outflow


@dataclass
class FlowResult:
    anomaly:          FlowAnomaly
    avg_speed:        float        # pixels/frame — proxy for movement speed
    direction_angle:  float        # degrees — 0=right, 90=up, 180=left
    flow_magnitude:   float        # overall movement intensity
    description:      str


class FlowAnalyzer:
    """
    Maintains a rolling 2-frame buffer per camera and
    computes optical flow to detect crowd movement anomalies.
    """

    def __init__(self):
        self._prev_frames:  dict[str, np.ndarray] = {}
        self._prev_gray:    dict[str, np.ndarray] = {}

        # Lucas-Kanade parameters
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        # Shi-Tomasi corner detection for tracking points
        self._feature_params = dict(
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
        )

    def analyze(self, camera_id: str, frame: np.ndarray) -> FlowResult:
        """
        Compute optical flow between the previous and current frame
        for a given camera and return anomaly classification.
        """
        if frame is None:
            return FlowResult(FlowAnomaly.NORMAL, 0, 0, 0, "No frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame for this camera — store and return normal
        if camera_id not in self._prev_gray:
            self._prev_gray[camera_id] = gray
            return FlowResult(FlowAnomaly.NORMAL, 0.0, 0.0, 0.0, "Initialising")

        prev_gray = self._prev_gray[camera_id]

        # Detect feature points in previous frame
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self._feature_params)
        if p0 is None or len(p0) == 0:
            self._prev_gray[camera_id] = gray
            return FlowResult(FlowAnomaly.NORMAL, 0.0, 0.0, 0.0, "No features")

        # Compute optical flow
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self._lk_params)

        # Keep only successfully tracked points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) == 0:
            self._prev_gray[camera_id] = gray
            return FlowResult(FlowAnomaly.NORMAL, 0.0, 0.0, 0.0, "No tracked points")

        # Compute motion vectors
        motion   = good_new - good_old
        speeds   = np.linalg.norm(motion, axis=1)
        avg_speed= float(np.mean(speeds))

        # Average direction vector
        avg_dx   = float(np.mean(motion[:, 0]))
        avg_dy   = float(np.mean(motion[:, 1]))
        angle    = float(np.degrees(np.arctan2(-avg_dy, avg_dx)) % 360)
        magnitude= float(np.sqrt(avg_dx**2 + avg_dy**2))

        self._prev_gray[camera_id] = gray

        return self._classify(avg_speed, angle, magnitude)

    def _classify(self, speed: float, angle: float, magnitude: float) -> FlowResult:
        """
        Classify flow into an anomaly category based on speed and direction.
        Thresholds below are empirical — tune for your venue and camera setup.
        """
        if speed < 0.5:
            return FlowResult(FlowAnomaly.BOTTLENECK, speed, angle, magnitude,
                              "Very low movement — possible bottleneck or crowd crush forming")

        if speed > 12.0:
            # High speed reverse flow → panic signature
            return FlowResult(FlowAnomaly.PANIC, speed, angle, magnitude,
                              "High-speed reverse flow — possible panic or stampede")

        if speed > 7.0:
            return FlowResult(FlowAnomaly.SURGE, speed, angle, magnitude,
                              "Rapid crowd movement — possible surge")

        if speed > 5.0:
            return FlowResult(FlowAnomaly.DISPERSAL, speed, angle, magnitude,
                              "Rapid dispersal — crowd moving out quickly")

        return FlowResult(FlowAnomaly.NORMAL, speed, angle, magnitude,
                          "Normal crowd flow")
