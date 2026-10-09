import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from backend.schemas.auth import (  # noqa: E402
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    SensorRecordResponse,
    SessionSummary,
    UserResponse,
)
from backend.schemas.sensor import SensorPacket  # noqa: E402
from backend.services import auth_service  # noqa: E402
from backend.storage import repository  # noqa: E402
from backend.storage.database import initialize_database  # noqa: E402
from backend.services.ml_service import ml_service  # noqa: E402


initialize_database()
bearer_scheme = HTTPBearer(auto_error=False)

app = FastAPI(
    title="NeuroSense API",
    description="Backend API for the NeuroSense attention estimation system",
    version="0.3.0",
)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class PredictionRequest(BaseModel):
    theta: float = Field(ge=0)
    alpha: float = Field(ge=0)
    beta: float = Field(ge=0)
    heart_rate: float = Field(gt=0)
    motion_level: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    state: str
    confidence: float
    attention_score: float
    focused_probability: float
    neutral_probability: float
    distracted_probability: float


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    token = credentials.credentials if credentials and credentials.scheme.lower() == "bearer" else None
    user = auth_service.current_user(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@app.get("/")
def root():
    return {"message": "NeuroSense Backend API", "status": "running"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", service="neurosense-backend", version="0.3.0")


@app.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest):
    try:
        auth_service.register_user(request.email, request.password, request.display_name)
        return AuthResponse(**auth_service.login_user(request.email, request.password))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/auth/login", response_model=AuthResponse)
def login(request: LoginRequest):
    try:
        return AuthResponse(**auth_service.login_user(request.email, request.password))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    _user: dict = Depends(get_current_user),
):
    auth_service.logout_user(credentials.credentials if credentials else None)
    return None


@app.get("/auth/me", response_model=UserResponse)
def me(user: dict = Depends(get_current_user)):
    return UserResponse(**user)


@app.get("/sessions", response_model=list[SessionSummary])
def sessions(user: dict = Depends(get_current_user)):
    return repository.list_user_sessions(user["id"])


@app.get("/sessions/{session_id}/records", response_model=list[SensorRecordResponse])
def session_records(session_id: str, user: dict = Depends(get_current_user)):
    if not repository.user_has_session(user["id"], session_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return repository.list_user_records(user["id"], session_id=session_id)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, _user: dict = Depends(get_current_user)):
    try:
        result = ml_service.predict(
            request.theta, request.alpha, request.beta, request.heart_rate, request.motion_level
        )
        return PredictionResponse(
            state=result["state"],
            confidence=result["confidence"],
            attention_score=result["attention_score"],
            focused_probability=result["focused_probability"],
            neutral_probability=result["neutral_probability"],
            distracted_probability=result["distracted_probability"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict-sensor")
def predict_sensor(request: SensorPacket, user: dict = Depends(get_current_user)):
    try:
        from backend.services.sensor_inference_service import sensor_inference_service

        result = sensor_inference_service.predict_from_sensor(
            eeg=request.eeg, heart_rate=request.heart_rate, motion_level=request.motion_level
        )
        record_id = repository.save_sensor_record(request, result, user_id=user["id"])
        result["record_id"] = record_id
        return result
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sensor prediction failed: {exc}") from exc
