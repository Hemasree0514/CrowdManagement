"""
backend/routes/dispatch.py
Autonomous alert dispatch engine.

Handles sending messages via:
  - SMS (Twilio)
  - WhatsApp (Twilio WhatsApp Business API)
  - In-app push (logged — connect Firebase for real push)
  - Radio/PTT (logged — integrate your radio gateway here)

All dispatch is fire-and-forget: the AI engine calls dispatch_alert()
with no human in the loop.
"""
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from backend.utils.logger import log
from config.settings import settings

try:
    from twilio.rest import Client as TwilioClient
    _TWILIO_AVAILABLE = bool(settings.twilio_account_sid and settings.twilio_auth_token)
except ImportError:
    _TWILIO_AVAILABLE = False
    log.warning("twilio_not_installed", hint="pip install twilio")


class Channel(str, Enum):
    SMS       = "SMS"
    WHATSAPP  = "WhatsApp"
    APP       = "App"
    RADIO     = "Radio"


@dataclass
class DispatchRecord:
    channel:   Channel
    to_unit:   str
    to_phone:  str
    message:   str
    sent_at:   str = field(default_factory=lambda: datetime.now().isoformat())
    success:   bool = True
    error:     str = ""


# In-memory log of all dispatched messages
dispatch_log: list[DispatchRecord] = []


# ── Message templates ──────────────────────────────────────────────────────

def msg_critical(zone_id, zone_name, count, capacity):
    return (
        f"🚨 ALERT Zone {zone_id} ({zone_name}): "
        f"crowd at {count}/{capacity}. "
        f"Redirect to adjacent zones. Dispatch 2 additional units immediately."
    )

def msg_warning(zone_id, zone_name, pct):
    return (
        f"⚠️ CAUTION Zone {zone_id} ({zone_name}): "
        f"occupancy at {pct:.0%}. Stay alert and monitor exits."
    )

def msg_bottleneck(zone_id, zone_name):
    return (
        f"🚧 BOTTLENECK Zone {zone_id} ({zone_name}). "
        f"Implement one-way flow. Block entry from Gate 2. Redirect via Gate 3."
    )

def msg_panic(zone_id, zone_name):
    return (
        f"🆘 URGENT: Abnormal crowd flow Zone {zone_id} ({zone_name}). "
        f"Verify situation. Do NOT block exits. Deploy calming protocol."
    )

def msg_vip(vip_name, vip_role, camera_id, zone_id):
    return (
        f"[CONFIDENTIAL] {vip_name} ({vip_role}) detected "
        f"at {camera_id} — Zone {zone_id}. "
        f"Coordinate escort protocol immediately."
    )

def msg_surge(zone_id, zone_name, count, capacity):
    return (
        f"📈 SURGE ALERT Zone {zone_id} ({zone_name}): "
        f"density at {count/capacity:.0%}. "
        f"Open Gate 5 immediately and deploy crowd barriers."
    )


# ── Core dispatch function ─────────────────────────────────────────────────

async def dispatch_alert(
    channel:  Channel,
    to_unit:  str,
    message:  str,
    phone:    str = "",
) -> DispatchRecord:
    """
    Send a message via the specified channel.
    Logs every dispatch regardless of success.
    """
    record = DispatchRecord(
        channel=channel,
        to_unit=to_unit,
        to_phone=phone,
        message=message,
    )

    try:
        if channel == Channel.SMS and phone:
            await _send_sms(phone, message, record)

        elif channel == Channel.WHATSAPP and phone:
            await _send_whatsapp(phone, message, record)

        elif channel in (Channel.APP, Channel.RADIO):
            # For App: integrate Firebase Admin SDK here
            # For Radio: integrate your PTT gateway SDK here
            log.info("dispatch_logged", channel=channel.value, unit=to_unit, message=message[:80])

    except Exception as e:
        record.success = False
        record.error   = str(e)
        log.error("dispatch_failed", channel=channel.value, unit=to_unit, error=str(e))

    dispatch_log.append(record)
    return record


async def _send_sms(phone: str, message: str, record: DispatchRecord):
    if not _TWILIO_AVAILABLE:
        log.info("sms_mock_sent", to=phone, message=message[:60])
        return

    def _send():
        client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
        client.messages.create(
            body=message,
            from_=settings.twilio_from_number,
            to=phone,
        )

    # Run in executor to avoid blocking async event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send)
    log.info("sms_sent", to=phone)


async def _send_whatsapp(phone: str, message: str, record: DispatchRecord):
    if not _TWILIO_AVAILABLE:
        log.info("whatsapp_mock_sent", to=phone, message=message[:60])
        return

    def _send():
        client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
        client.messages.create(
            body=message,
            from_=settings.twilio_whatsapp_from,
            to=f"whatsapp:{phone}",
        )

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send)
    log.info("whatsapp_sent", to=phone)
