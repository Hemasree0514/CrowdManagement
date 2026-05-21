"""
backend/main.py
FastAPI application entry point.

Starts:
  - FastAPI REST API (HTTP) on port 8000
  - WebSocket server on port 8765
  - Camera streams (background threads)
  - AI pipeline loop (async)

Run with:
    python backend/main.py
or:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from backend.utils.zones import init_zones
from backend.utils.camera import CameraManager
from backend.models.crowd_detector import CrowdDetector
from backend.models.vip_recognizer import VipRecognizer
from backend.models.flow_analyzer import FlowAnalyzer
from backend.routes.alerts import router as alert_router
from backend.websocket_server import start_websocket_server
from backend.utils.logger import log


# ── Camera URL map (from .env) ─────────────────────────────────────────────
CAMERA_URLS = {
    "CAM_01": settings.cam_01,
    "CAM_02": settings.cam_02,
    "CAM_03": settings.cam_03,
    "CAM_04": settings.cam_04,
    "CAM_05": settings.cam_05,
    "CAM_06": settings.cam_06,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise zones, cameras, AI models.
    Shutdown: stop camera streams.
    """
    log.info("starting_crowd_ai_system")

    # 1. Initialise zone registry
    zones = init_zones()
    log.info("zones_loaded", count=len(zones))

    # 2. Start camera streams
    camera_manager = CameraManager(CAMERA_URLS)
    camera_manager.start()

    # 3. Load AI models
    crowd_detector = CrowdDetector(model_name="yolov8s.pt", confidence=0.4)
    vip_recognizer = VipRecognizer()
    flow_analyzer  = FlowAnalyzer()

    # 4. Start WebSocket server + AI pipeline in background
    asyncio.create_task(
        start_websocket_server(camera_manager, crowd_detector, vip_recognizer, flow_analyzer)
    )

    log.info("system_ready", ws_port=settings.websocket_port, api_port=settings.port)
    yield

    # Shutdown
    camera_manager.stop()
    log.info("system_shutdown")


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Crowd Management System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API routes
app.include_router(alert_router)

# Serve the frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
