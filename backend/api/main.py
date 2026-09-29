import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from backend.services.ml_service import ml_service


app = FastAPI(
    title="NeuroSense API",
    description="Backend API for the NeuroSense attention estimation system",
    version="0.2.0"
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


@app.get("/")
def root():
    return {
        "message": "NeuroSense Backend API",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        service="neurosense-backend",
        version="0.2.0"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = ml_service.predict(
            request.theta,
            request.alpha,
            request.beta,
            request.heart_rate,
            request.motion_level
        )

        return PredictionResponse(
            state=result["state"],
            confidence=result["confidence"],
            attention_score=result["attention_score"],
            focused_probability=result["focused_probability"],
            neutral_probability=result["neutral_probability"],
            distracted_probability=result["distracted_probability"]
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        )
