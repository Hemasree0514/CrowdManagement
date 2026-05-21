"""
scripts/test_cameras.py
Test all configured camera RTSP streams.
Run this before starting the main system to verify all cameras are reachable.

Usage:
    python scripts/test_cameras.py
"""
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings

CAMERA_URLS = {
    "CAM_01": settings.cam_01,
    "CAM_02": settings.cam_02,
    "CAM_03": settings.cam_03,
    "CAM_04": settings.cam_04,
    "CAM_05": settings.cam_05,
    "CAM_06": settings.cam_06,
}


def test_camera(cam_id: str, url: str) -> bool:
    if not url:
        print(f"  [{cam_id}] ⚠️  No URL configured — skipping")
        return False

    print(f"  [{cam_id}] Testing: {url[:40]}...")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"  [{cam_id}] ❌ FAILED — could not open stream")
        return False

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        h, w = frame.shape[:2]
        print(f"  [{cam_id}] ✅ OK — frame: {w}x{h}")
        return True
    else:
        print(f"  [{cam_id}] ❌ FAILED — stream opened but no frame received")
        return False


def main():
    print("\n🎥 AI Crowd Management — Camera connectivity test\n")
    results = {}
    for cam_id, url in CAMERA_URLS.items():
        results[cam_id] = test_camera(cam_id, url)

    ok    = sum(results.values())
    total = len([u for u in CAMERA_URLS.values() if u])
    print(f"\n{'─'*40}")
    print(f"Result: {ok}/{total} cameras reachable")
    if ok < total:
        print("Fix camera URLs in your .env file before starting the system.")
    else:
        print("All cameras OK — ready to start.")


if __name__ == "__main__":
    main()
