"""User-scoped dashboard persistence helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from backend.schemas.sensor import SensorPacket
from backend.storage import repository


def save_live_reading(user: dict, sensor_data: dict, result: dict) -> int:
    session_id = sensor_data.get("session_id") or "dashboard-" + uuid4().hex
    packet = SensorPacket(
        session_id=session_id,
        subject_id=f"dashboard-user-{user['id']}",
        device_id=str(sensor_data.get("mode", "simulated")),
        timestamp=datetime.now(timezone.utc),
        sampling_rate=256,
        eeg=[float(value) for value in sensor_data["eeg"]],
        heart_rate=float(sensor_data["heart_rate"]),
        motion_level=float(sensor_data["motion_level"]),
    )
    return repository.save_sensor_record(packet, result, user_id=user["id"])


def list_sessions(user: dict) -> list[dict]:
    return repository.list_user_sessions(user["id"])


def list_records(user: dict, session_id: str | None = None, limit: int = 500) -> list[dict]:
    return repository.list_user_records(user["id"], session_id=session_id, limit=limit)
