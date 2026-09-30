import json

from backend.storage.database import get_connection


def save_sensor_record(packet, result):
    prediction = result.get("prediction") or {}
    signal_quality = result.get("signal_quality") or {}

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sensor_records (
                session_id,
                subject_id,
                device_id,
                timestamp,
                sampling_rate,
                eeg_json,
                heart_rate,
                motion_level,
                prediction_allowed,
                signal_status,
                state,
                confidence,
                attention_score,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                packet.session_id,
                packet.subject_id,
                packet.device_id,
                packet.timestamp.isoformat(),
                packet.sampling_rate,
                json.dumps(packet.eeg),
                packet.heart_rate,
                packet.motion_level,
                int(result.get("prediction_allowed", False)),
                signal_quality.get("status"),
                prediction.get("state"),
                prediction.get("confidence"),
                prediction.get("attention_score"),
                prediction.get("model_version"),
            ),
        )

        conn.commit()
        return cursor.lastrowid
