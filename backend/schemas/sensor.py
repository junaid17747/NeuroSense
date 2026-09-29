from datetime import datetime
from pydantic import BaseModel, Field


class SensorPacket(BaseModel):
    session_id: str
    subject_id: str
    device_id: str

    timestamp: datetime
    sampling_rate: int = Field(default=256, gt=0)

    eeg: list[float] = Field(min_length=1024, max_length=1024)

    heart_rate: float = Field(gt=0)
    motion_level: float = Field(ge=0, le=1)


class SensorPredictionResponse(BaseModel):
    session_id: str
    timestamp: datetime
    prediction_allowed: bool
    signal_quality: dict
    features: dict | None = None
    prediction: dict | None = None
