"""
backend/websocket_server.py
WebSocket server — the real-time bridge between the AI pipeline and the frontend.

Every ~1 second it:
  1. Reads latest frames from all cameras
  2. Runs crowd detection (YOLO) on each frame
  3. Runs face recognition (InsightFace) on each frame
  4. Runs flow analysis (optical flow) on each frame
  5. Evaluates thresholds → fires autonomous alerts
  6. Broadcasts a JSON state update to ALL connected browser clients

Frontend connects with:
    const ws = new WebSocket('ws://localhost:8765');
    ws.onmessage = (e) => { const data = JSON.parse(e.data); updateUI(data); }
"""
import asyncio
import json
import websockets
from datetime import datetime
from backend.utils.camera import CameraManager
from backend.utils.zones import get_zones, get_zone
from backend.models.crowd_detector import CrowdDetector
from backend.models.vip_recognizer import VipRecognizer
from backend.models.flow_analyzer import FlowAnalyzer, FlowAnomaly
from backend.routes.dispatch import (
    dispatch_alert, Channel,
    msg_critical, msg_warning, msg_bottleneck, msg_panic, msg_vip, msg_surge,
    dispatch_log,
)
from config.settings import settings
from backend.utils.logger import log

# ── Alert cooldown: prevent re-alerting the same zone too quickly ──────────
ALERT_COOLDOWN_TICKS = 10   # ~10 seconds at 1 tick/s

# In-memory alert event log (separate from dispatch log)
alert_events: list[dict] = []

# Connected WebSocket clients
connected_clients: set = set()


async def ai_pipeline_loop(
    camera_manager:  CameraManager,
    crowd_detector:  CrowdDetector,
    vip_recognizer:  VipRecognizer,
    flow_analyzer:   FlowAnalyzer,
):
    """
    Main AI loop. Runs forever.
    One tick per second — process all cameras, evaluate thresholds, broadcast.
    """
    zones        = get_zones()
    cooldowns    = {z_id: 0 for z_id in zones}
    vip_events   = []
    tick         = 0

    while True:
        tick += 1
        zone_updates = {}

        for zone_id, zone in zones.items():
            frame = camera_manager.get_frame(zone.camera_id)

            # ── 1. Crowd counting ────────────────────────────────────────
            detection = crowd_detector.detect(frame)
            zone.update(detection.count)

            # ── 2. Flow analysis ─────────────────────────────────────────
            flow = flow_analyzer.analyze(zone.camera_id, frame)

            # ── 3. VIP face recognition ──────────────────────────────────
            vip_matches = vip_recognizer.recognize(frame)
            for match in vip_matches:
                event = {
                    "type":       "vip",
                    "vip_id":     match.vip_id,
                    "name":       match.name,
                    "role":       match.role,
                    "rank":       match.rank,
                    "confidence": match.confidence,
                    "camera_id":  zone.camera_id,
                    "zone_id":    zone_id,
                    "zone_name":  zone.name,
                    "time":       datetime.now().strftime("%H:%M:%S"),
                }
                vip_events.append(event)
                alert_events.append({**event, "title": f"VIP detected — {match.name}"})
                log.info("vip_detected", name=match.name, confidence=match.confidence, zone=zone_id)

                # Autonomous dispatch — no human needed
                phone = settings.phone_for_unit(match.notify_unit)
                await dispatch_alert(Channel.APP,  match.notify_unit,
                                     msg_vip(match.name, match.role, zone.camera_id, zone_id), phone)
                await dispatch_alert(Channel.SMS,  "Protocol Officer",
                                     f"VIP arrival confirmed: {match.name}. Clear pathway to Zone F.",
                                     settings.protocol_officer_phone)

            # ── 4. Threshold evaluation + autonomous alerting ────────────
            lvl = zone.level
            if cooldowns[zone_id] > 0:
                cooldowns[zone_id] -= 1
            else:
                if lvl == "critical":
                    _add_alert("critical",
                               f"Critical — Zone {zone_id}: {zone.name}",
                               f"Density {zone.current_count}/{zone.capacity}.")
                    phone = settings.phone_for_unit(zone.security_unit)
                    await dispatch_alert(Channel.SMS,   zone.security_unit,
                                         msg_critical(zone_id, zone.name, zone.current_count, zone.capacity), phone)
                    await dispatch_alert(Channel.RADIO, "Command Centre",
                                         f"Zone {zone_id} critical. Activating overflow protocol.")
                    cooldowns[zone_id] = ALERT_COOLDOWN_TICKS

                elif lvl == "warning":
                    _add_alert("warning",
                               f"Caution — Zone {zone_id}: {zone.name}",
                               f"Density {zone.current_count}/{zone.capacity}.")
                    phone = settings.phone_for_unit(zone.security_unit)
                    await dispatch_alert(Channel.SMS, zone.security_unit,
                                         msg_warning(zone_id, zone.name, zone.occupancy_pct), phone)
                    cooldowns[zone_id] = ALERT_COOLDOWN_TICKS

            # Bottleneck detection from flow analyzer
            if flow.anomaly == FlowAnomaly.BOTTLENECK and lvl != "safe" and cooldowns[zone_id] == 0:
                _add_alert("warning",
                           f"Bottleneck — Zone {zone_id}",
                           flow.description)
                phone = settings.phone_for_unit(zone.security_unit)
                await dispatch_alert(Channel.SMS, zone.security_unit,
                                     msg_bottleneck(zone_id, zone.name), phone)
                cooldowns[zone_id] = ALERT_COOLDOWN_TICKS

            if flow.anomaly == FlowAnomaly.PANIC and cooldowns[zone_id] == 0:
                _add_alert("critical", f"Panic flow — Zone {zone_id}", flow.description)
                phone = settings.phone_for_unit(zone.security_unit)
                await dispatch_alert(Channel.SMS,  zone.security_unit,
                                     msg_panic(zone_id, zone.name), phone)
                await dispatch_alert(Channel.SMS,  "Medical team",
                                     f"Crowd anomaly Zone {zone_id}. Position at east exit.",
                                     settings.medical_team_phone)
                cooldowns[zone_id] = ALERT_COOLDOWN_TICKS

            zone_updates[zone_id] = {
                "count":      zone.current_count,
                "capacity":   zone.capacity,
                "pct":        round(zone.occupancy_pct * 100, 1),
                "level":      lvl,
                "trend":      zone.trend,
                "flow":       flow.anomaly.value,
                "flow_speed": round(flow.avg_speed, 2),
            }

        # ── 5. Broadcast to all connected frontend clients ───────────────
        if connected_clients:
            payload = json.dumps({
                "tick":       tick,
                "zones":      zone_updates,
                "alerts":     alert_events[-20:],
                "messages":   [
                    {"channel": r.channel.value, "to": r.to_unit,
                     "text": r.message, "time": r.sent_at[-8:]}
                    for r in dispatch_log[-30:]
                ],
                "vip_events": vip_events[-20:],
                "total_crowd":sum(z.current_count for z in zones.values()),
            })
            dead = set()
            for ws in connected_clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            connected_clients -= dead

        await asyncio.sleep(1.0)


def _add_alert(level: str, title: str, body: str):
    alert_events.append({
        "type":  level,
        "title": title,
        "body":  body,
        "time":  datetime.now().strftime("%H:%M:%S"),
    })
    log.info("alert_fired", level=level, title=title)


async def ws_handler(websocket, path=""):
    """Handle new WebSocket connection from the frontend."""
    connected_clients.add(websocket)
    log.info("client_connected", total=len(connected_clients))
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        log.info("client_disconnected", total=len(connected_clients))


async def start_websocket_server(
    camera_manager: CameraManager,
    crowd_detector: CrowdDetector,
    vip_recognizer: VipRecognizer,
    flow_analyzer:  FlowAnalyzer,
):
    """Start the WebSocket server and AI pipeline concurrently."""
    server = await websockets.serve(ws_handler, "0.0.0.0", settings.websocket_port)
    log.info("websocket_server_started", port=settings.websocket_port)

    await asyncio.gather(
        server.wait_closed(),
        ai_pipeline_loop(camera_manager, crowd_detector, vip_recognizer, flow_analyzer),
    )
