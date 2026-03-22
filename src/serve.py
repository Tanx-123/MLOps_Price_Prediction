"""
FastAPI app for serving rent predictions.

Endpoints:
    GET  /health       — Health check
    GET  /model/info   — Model metadata and metrics
    POST /predict      — Predict rent from features

Usage:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""
import os
import json
import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core_utils import load_config, ensure_local_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Request / Response schemas ───────────────────────────────────────

class RentInput(BaseModel):
    BHK: int = Field(..., ge=1, le=10)
    Size: int = Field(..., gt=0)
    Bathroom: int = Field(..., ge=1, le=10)
    floor_num: int = Field(..., ge=-2)
    total_floors: int = Field(..., ge=1)
    Area_Type: str
    City: str
    Furnishing_Status: str
    Tenant_Preferred: str
    Area_Locality: str


class RentOutput(BaseModel):
    predicted_rent: float
    currency: str = "INR"


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False


class ModelInfoResponse(BaseModel):
    model_type: str
    metrics: dict
    features_count: int


# ── Global state (populated on startup) ──────────────────────────────

model = None
preprocessor = None
target_encoding_maps = None
metrics = None
config = None


def _load_joblib(artifacts_dir, bucket, prefix, filename):
    """Load a joblib file, downloading from S3 if it doesn't exist locally."""
    path = os.path.join(artifacts_dir, filename)
    if not os.path.exists(path):
        if not ensure_local_file(path, bucket, f"{prefix}/{filename}"):
            raise RuntimeError(f"Failed to load artifact: {filename}")
    return joblib.load(path)


def _load_json(artifacts_dir, *filenames):
    """Try loading the first JSON file that exists from the list."""
    for name in filenames:
        path = os.path.join(artifacts_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


async def _startup():
    global model, preprocessor, target_encoding_maps, metrics, config

    config = load_config()
    s3_cfg = config["s3"]
    artifacts_dir = config["data"]["artifacts_path"]
    bucket, prefix = s3_cfg["bucket"], s3_cfg["artifacts_prefix"]

    model = _load_joblib(artifacts_dir, bucket, prefix, "best_model.joblib")
    preprocessor = _load_joblib(artifacts_dir, bucket, prefix, "preprocessor.joblib")
    target_encoding_maps = _load_joblib(artifacts_dir, bucket, prefix, "target_encoding_maps.joblib")
    metrics = _load_json(artifacts_dir, "ensemble_results.json", "metrics.json")

    logger.info(f"Loaded model: {model.__class__.__name__}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield


# ── App setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="Rent Predictor API",
    description="ML-powered rent prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    if os.path.isdir("static"):
        return RedirectResponse(url="/static/index.html")
    return {"message": "Rent Predictor API — visit /docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return ModelInfoResponse(
        model_type=config["model"]["type"],
        metrics=metrics or {},
        features_count=getattr(model, "n_features_in_", 0),
    )


@app.post("/predict", response_model=RentOutput)
async def predict(input_data: RentInput):
    if model is None or preprocessor is None:
        raise HTTPException(503, "Model not loaded")

    try:
        # Build a single-row DataFrame matching the training column names
        df = pd.DataFrame([{
            "BHK": input_data.BHK,
            "Size": input_data.Size,
            "Bathroom": input_data.Bathroom,
            "floor_num": input_data.floor_num,
            "total_floors": input_data.total_floors,
            "Area Type": input_data.Area_Type,
            "City": input_data.City,
            "Furnishing Status": input_data.Furnishing_Status,
            "Tenant Preferred": input_data.Tenant_Preferred,
            "Area Locality": input_data.Area_Locality.lower(),
        }])

        # Apply target encoding (same logic as training)
        if target_encoding_maps:
            for col, enc_map in target_encoding_maps.items():
                if col in df.columns:
                    fallback = np.mean(list(enc_map.values()))
                    df[col] = df[col].map(enc_map).fillna(fallback)

        X = preprocessor.transform(df)
        pred = float(model.predict(X)[0])
        return RentOutput(predicted_rent=round(max(pred, 0), 2))

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(400, f"Prediction error: {e}")
