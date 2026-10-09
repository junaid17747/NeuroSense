"""Persistence helpers for legacy sensor data and authenticated user data."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.storage.database import get_connection


def _utc_iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


def create_user(email: str, password_hash: str, display_name: str) -> dict:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, display_name) VALUES (?, ?, ?)",
            (email, password_hash, display_name),
        )
        row = conn.execute(
            "SELECT id, email, display_name, created_at FROM users WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        conn.commit()
        return dict(row)


def get_user_by_email(email: str):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


def get_user_by_id(user_id: int):
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, email, display_name, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()


def update_password(user_id: int, password_hash: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
        conn.commit()


def touch_login(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (_utc_iso(), user_id))
        conn.commit()


def create_auth_session(token_hash: str, user_id: int, expires_at: datetime) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO auth_sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, _utc_iso(expires_at)),
        )
        conn.commit()


def get_user_for_session(token_hash: str):
    now = _utc_iso()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email, u.display_name, u.created_at
            FROM auth_sessions AS s
            JOIN users AS u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ?
            """,
            (token_hash, now),
        ).fetchone()
        if row:
            conn.execute("UPDATE auth_sessions SET last_seen_at = ? WHERE token_hash = ?", (now, token_hash))
            conn.commit()
        return row


def revoke_auth_session(token_hash: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ?", (_utc_iso(), token_hash))
        conn.commit()


def ensure_eeg_session(user_id: int, session_id: str, subject_id: str, device_id: str) -> None:
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT user_id FROM eeg_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if existing and int(existing["user_id"]) != int(user_id):
            raise PermissionError("This EEG session belongs to another account.")
        conn.execute(
            """
            INSERT INTO eeg_sessions (session_id, user_id, subject_id, device_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET last_seen_at = excluded.last_seen_at
            """,
            (session_id, user_id, subject_id, device_id),
        )
        conn.commit()


def save_sensor_record(packet, result, user_id: int | None = None):
    prediction = result.get("prediction") or {}
    signal_quality = result.get("signal_quality") or {}
    if user_id is not None:
        ensure_eeg_session(user_id, packet.session_id, packet.subject_id, packet.device_id)

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sensor_records (
                session_id, subject_id, device_id, timestamp, sampling_rate, eeg_json,
                heart_rate, motion_level, prediction_allowed, signal_status, state,
                confidence, attention_score, model_version, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                packet.session_id, packet.subject_id, packet.device_id, packet.timestamp.isoformat(),
                packet.sampling_rate, json.dumps(packet.eeg), packet.heart_rate, packet.motion_level,
                int(result.get("prediction_allowed", False)), signal_quality.get("status"),
                prediction.get("state"), prediction.get("confidence"), prediction.get("attention_score"),
                prediction.get("model_version"), user_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def list_user_sessions(user_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT session_id, MAX(device_id) AS device_id, MAX(subject_id) AS subject_id,
                   MIN(timestamp) AS first_seen, MAX(timestamp) AS last_seen, COUNT(*) AS record_count
            FROM sensor_records WHERE user_id = ? GROUP BY session_id ORDER BY last_seen DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def list_user_records(user_id: int, session_id: str | None = None, limit: int = 500) -> list[dict]:
    limit = max(1, min(int(limit), 5000))
    query = """
        SELECT id, session_id, subject_id, device_id, timestamp, heart_rate, motion_level,
               prediction_allowed, signal_status, state, confidence, attention_score,
               model_version, created_at
        FROM sensor_records WHERE user_id = ?
    """
    params: list[object] = [user_id]
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def user_has_session(user_id: int, session_id: str) -> bool:
    with get_connection() as conn:
        return bool(conn.execute(
            "SELECT 1 FROM sensor_records WHERE user_id = ? AND session_id = ? LIMIT 1",
            (user_id, session_id),
        ).fetchone())
