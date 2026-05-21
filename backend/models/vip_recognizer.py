"""
backend/models/vip_recognizer.py
VIP / dignitary face recognition using InsightFace.

Workflow:
  1. At startup: load VIP images → extract embeddings → store in memory
  2. Per frame:  detect faces → extract embeddings → compare with VIP DB
  3. Match above confidence threshold → fire VIP detection event

Install:
    pip install insightface onnxruntime
"""
import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from backend.utils.logger import log
from config.settings import settings

try:
    import insightface
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    log.warning("insightface_not_installed", hint="pip install insightface onnxruntime")


@dataclass
class VipMatch:
    vip_id:     str
    name:       str
    role:       str
    rank:       str
    confidence: float
    bbox:       list        # [x1, y1, x2, y2]
    notify_unit:str
    escort_required: bool


class VipRecognizer:
    """
    Loads registered VIP face embeddings at startup and
    compares each detected face against the database each frame.
    """

    def __init__(self):
        self.app         = None
        self.embeddings  = []   # list of { vip_data, embedding }
        self.vip_db      = settings.vip_db
        self.images_dir  = Path(settings.vip_images_dir)
        self.threshold   = settings.vip_confidence_threshold

        if _INSIGHTFACE_AVAILABLE:
            self._init_model()
            self._load_vip_embeddings()
        else:
            log.warning("vip_recognizer_running_in_mock_mode")

    def _init_model(self):
        self.app = FaceAnalysis(
            name=settings.face_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        log.info("insightface_model_loaded", model=settings.face_model)

    def _load_vip_embeddings(self):
        """
        Read each VIP's reference image, compute its embedding,
        and store alongside metadata.
        """
        import cv2
        loaded = 0
        for vip in self.vip_db:
            img_path = self.images_dir / vip["image_file"]
            if not img_path.exists():
                log.warning("vip_image_not_found", vip=vip["id"], path=str(img_path))
                continue
            img  = cv2.imread(str(img_path))
            faces = self.app.get(img)
            if not faces:
                log.warning("no_face_in_vip_image", vip=vip["id"])
                continue
            embedding = faces[0].embedding
            self.embeddings.append({"vip": vip, "embedding": embedding})
            loaded += 1
        log.info("vip_embeddings_loaded", count=loaded)

    def recognize(self, frame: np.ndarray) -> list[VipMatch]:
        """
        Run face recognition on a frame.
        Returns a list of VipMatch for every matched VIP face.
        """
        if frame is None:
            return []

        if self.app is None:
            return self._mock_recognize()

        faces = self.app.get(frame)
        matches = []
        for face in faces:
            match = self._compare_embedding(face.embedding, face.bbox.astype(int).tolist())
            if match:
                matches.append(match)
        return matches

    def _compare_embedding(self, embedding: np.ndarray, bbox: list) -> VipMatch | None:
        """Cosine similarity comparison against all registered VIPs."""
        best_score = 0.0
        best_vip   = None
        for entry in self.embeddings:
            score = self._cosine_similarity(embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_vip   = entry["vip"]

        if best_vip and best_score >= self.threshold:
            return VipMatch(
                vip_id=best_vip["id"],
                name=best_vip["name"],
                role=best_vip["role"],
                rank=best_vip["rank"],
                confidence=round(float(best_score), 4),
                bbox=bbox,
                notify_unit=best_vip["notify_unit"],
                escort_required=best_vip["escort_required"],
            )
        return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_n = a / (np.linalg.norm(a) + 1e-6)
        b_n = b / (np.linalg.norm(b) + 1e-6)
        return float(np.dot(a_n, b_n))

    def _mock_recognize(self) -> list[VipMatch]:
        """Mock — fires randomly 2% of the time for dev/testing."""
        if np.random.random() > 0.02 or not self.vip_db:
            return []
        vip = np.random.choice(self.vip_db)
        return [VipMatch(
            vip_id=vip["id"],
            name=vip["name"],
            role=vip["role"],
            rank=vip["rank"],
            confidence=round(np.random.uniform(0.92, 0.99), 4),
            bbox=[100, 100, 200, 250],
            notify_unit=vip["notify_unit"],
            escort_required=vip["escort_required"],
        )]
