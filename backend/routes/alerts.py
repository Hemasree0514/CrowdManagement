"""
backend/routes/alerts.py
REST API endpoints for alert history and zone status.
Mounted at /api/alerts and /api/zones in main.py.
"""
from fastapi import APIRouter
from backend.utils.zones import get_zones
from backend.routes.dispatch import dispatch_log

router = APIRouter(prefix="/api")


@router.get("/zones")
def list_zones():
    """Return current state of all zones."""
    zones = get_zones()
    return [
        {
            "id":           z.id,
            "name":         z.name,
            "capacity":     z.capacity,
            "current_count":z.current_count,
            "occupancy_pct":round(z.occupancy_pct, 3),
            "level":        z.level,
            "trend":        z.trend,
            "security_unit":z.security_unit,
            "camera_id":    z.camera_id,
        }
        for z in zones.values()
    ]


@router.get("/zones/{zone_id}")
def get_zone(zone_id: str):
    """Return state of a single zone."""
    zones = get_zones()
    z = zones.get(zone_id.upper())
    if not z:
        return {"error": "Zone not found"}
    return {
        "id":           z.id,
        "name":         z.name,
        "capacity":     z.capacity,
        "current_count":z.current_count,
        "occupancy_pct":round(z.occupancy_pct, 3),
        "level":        z.level,
        "trend":        z.trend,
        "security_unit":z.security_unit,
        "adjacent_zones": z.adjacent_zones,
        "overflow_gates": z.overflow_gates,
    }


@router.get("/messages")
def list_messages(limit: int = 50):
    """Return the last N dispatched messages."""
    return [
        {
            "channel":  r.channel.value,
            "to_unit":  r.to_unit,
            "message":  r.message,
            "sent_at":  r.sent_at,
            "success":  r.success,
        }
        for r in reversed(dispatch_log[-limit:])
    ]


@router.get("/health")
def health():
    return {"status": "ok"}
