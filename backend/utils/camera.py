"""
backend/utils/camera.py
RTSP camera stream manager.
Opens each camera in a background thread and provides the latest
frame to the AI pipeline on demand (non-blocking read).

Usage:
    manager = CameraManager(camera_urls)
    manager.start()
    frame = manager.get_frame("CAM_01")   # returns latest BGR numpy array
    manager.stop()
"""
import cv2
import threading
import time
from backend.utils.logger import log


class CameraStream:
    """Captures frames from a single RTSP stream in a background thread."""

    def __init__(self, camera_id: str, url: str):
        self.camera_id = camera_id
        self.url       = url
        self.frame     = None
        self.running   = False
        self._lock     = threading.Lock()
        self._thread   = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info("camera_stream_started", camera=self.camera_id, url=self.url)

    def _capture_loop(self):
        while self.running:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                log.warning("camera_connect_failed", camera=self.camera_id)
                time.sleep(5)   # retry after 5 s
                continue

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    log.warning("camera_frame_read_failed", camera=self.camera_id)
                    break
                with self._lock:
                    self.frame = frame   # always keep the latest frame only

            cap.release()
            if self.running:
                time.sleep(2)  # reconnect delay

    def get_frame(self):
        """Returns the most recent frame as a BGR numpy array, or None."""
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)


class CameraManager:
    """
    Manages multiple CameraStream instances.
    Maps camera_id → CameraStream.

    To use in production:
        urls = {
            "CAM_01": settings.cam_01,
            "CAM_02": settings.cam_02,
            ...
        }
        manager = CameraManager(urls)
        manager.start()
    """

    def __init__(self, camera_urls: dict[str, str]):
        self.streams: dict[str, CameraStream] = {
            cam_id: CameraStream(cam_id, url)
            for cam_id, url in camera_urls.items()
            if url  # skip empty URLs
        }

    def start(self):
        for stream in self.streams.values():
            stream.start()

    def get_frame(self, camera_id: str):
        stream = self.streams.get(camera_id)
        return stream.get_frame() if stream else None

    def stop(self):
        for stream in self.streams.values():
            stream.stop()
        log.info("all_camera_streams_stopped")
