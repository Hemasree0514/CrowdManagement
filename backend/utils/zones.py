"""
backend/utils/zones.py
Zone state management and threshold evaluation.
Loaded once at startup; updated every AI tick.
"""
from dataclasses import dataclass, field
from typing import Literal
from config.settings import settings


Level = Literal["safe", "warning", "critical"]


@dataclass
class ZoneState:
    id:           str
    name:         str
    capacity:     int
    security_unit:str
    camera_id:    str
    adjacent_zones: list
    overflow_gates: list
    current_count:  int = 0
    prev_count:     int = 0
    alert_cooldown: int = 0   # ticks before next alert for this zone

    @property
    def occupancy_pct(self) -> float:
        return self.current_count / self.capacity

    @property
    def level(self) -> Level:
        pct = self.occupancy_pct
        if pct >= settings.density_critical_pct:
            return "critical"
        if pct >= settings.density_warning_pct:
            return "warning"
        return "safe"

    @property
    def trend(self) -> Literal["rising", "falling", "stable"]:
        diff = self.current_count - self.prev_count
        if diff > 8:   return "rising"
        if diff < -8:  return "falling"
        return "stable"

    def update(self, new_count: int):
        self.prev_count    = self.current_count
        self.current_count = new_count
        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1


# ── Singleton zone registry ────────────────────────────────────────────
_zone_registry: dict[str, ZoneState] = {}


def init_zones() -> dict[str, ZoneState]:
    """Call once at startup to build the zone registry."""
    global _zone_registry
    for z in settings.zones:
        _zone_registry[z["id"]] = ZoneState(
            id=z["id"],
            name=z["name"],
            capacity=z["capacity"],
            security_unit=z["security_unit"],
            camera_id=z["camera_id"],
            adjacent_zones=z.get("adjacent_zones", []),
            overflow_gates=z.get("overflow_gates", []),
        )
    return _zone_registry


def get_zones() -> dict[str, ZoneState]:
    return _zone_registry


def get_zone(zone_id: str) -> ZoneState | None:
    return _zone_registry.get(zone_id)
