from datetime import datetime

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    display_name: str = Field(min_length=1, max_length=80)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class UserResponse(BaseModel):
    id: int
    email: str
    display_name: str
    created_at: datetime | str | None = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class SessionSummary(BaseModel):
    session_id: str
    device_id: str | None = None
    subject_id: str | None = None
    first_seen: str | None = None
    last_seen: str | None = None
    record_count: int


class SensorRecordResponse(BaseModel):
    id: int
    session_id: str
    subject_id: str
    device_id: str
    timestamp: str
    heart_rate: float
    motion_level: float
    prediction_allowed: bool | None
    signal_status: str | None
    state: str | None
    confidence: float | None
    attention_score: float | None
    model_version: str | None
    created_at: str | None
