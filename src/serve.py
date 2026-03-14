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

import yaml
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.core_utils import load_config, ensure_local_file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── Pydantic models ──────────────────────────────────────────────────

class RentInput(BaseModel):
    """Input features for rent prediction."""
    BHK: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    Size: int = Field(..., gt=0, description="Size in sqft")
    Bathroom: int = Field(..., ge=1, le=10, description="Number of bathrooms")
    floor_num: int = Field(..., ge=-2, description="Floor number (-2=lower basement, -1=upper basement, 0=ground)")
    total_floors: int = Field(..., ge=1, description="Total floors in building")
    Area_Type: str = Field(..., description="Type of area (e.g., Super Area, Carpet Area, Built Area)")
    City: str = Field(..., description="City name")
    Furnishing_Status: str = Field(..., description="Furnishing status (Furnished, Semi-Furnished, Unfurnished)")
    Tenant_Preferred: str = Field(..., description="Tenant preference (Bachelors, Bachelors/Family, Family)")
    Area_Locality: str = Field(..., description="Area/Locality name")


class RentOutput(BaseModel):
    """Output prediction response."""
    predicted_rent: float = Field(..., description="Predicted monthly rent in INR")
    currency: str = "INR"


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False


class ModelInfoResponse(BaseModel):
    model_type: str
    metrics: dict
    features_count: int


# ── Load config and artifacts ────────────────────────────────────────


def ensure_artifact(local_path, bucket, s3_key):
    """Load artifact from local path, downloading from S3 if needed."""
    if not os.path.exists(local_path):
        logger.info(f"Artifact not found locally, downloading from S3: {s3_key}")
        success = ensure_local_file(local_path, bucket, s3_key)
        if not success:
            raise RuntimeError(f"Failed to download artifact: {s3_key}")
    return local_path


# ── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(
    title="Rent Predictor API",
    description="ML-powered API to predict house rent based on property features",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the frontend."""
    return RedirectResponse(url="/static/index.html")

# Global state
model = None
preprocessor = None
target_encoding_maps = None
metrics = None
config = None


@app.on_event("startup")
async def startup():
    """Load best model, preprocessor, and encoding maps on startup."""
    global model, preprocessor, target_encoding_maps, metrics, config

    config = load_config()
    s3_config = config["s3"]
    artifacts_dir = config["data"]["artifacts_path"]
    bucket = s3_config["bucket"]
    prefix = s3_config["artifacts_prefix"]

    try:
        # Load best model (optimized ensemble or single model)
        model_path = ensure_artifact(
            os.path.join(artifacts_dir, "best_model.joblib"),
            bucket, f"{prefix}/best_model.joblib",
        )
        model = joblib.load(model_path)
        logger.info("Best model loaded successfully.")

        # Load preprocessor
        preprocessor_path = ensure_artifact(
            os.path.join(artifacts_dir, "preprocessor.joblib"),
            bucket, f"{prefix}/preprocessor.joblib",
        )
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded successfully.")

        # Load target encoding maps
        encoding_path = ensure_artifact(
            os.path.join(artifacts_dir, "target_encoding_maps.joblib"),
            bucket, f"{prefix}/target_encoding_maps.joblib",
        )
        target_encoding_maps = joblib.load(encoding_path)
        logger.info("Target encoding maps loaded successfully.")

        # Load model metrics and performance information
        metrics_path = os.path.join(artifacts_dir, "ensemble_results.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            logger.info("Model performance metrics loaded successfully.")
        else:
            # Fallback to individual model metrics
            metrics_path = os.path.join(artifacts_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                logger.info("Model metrics loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata and metrics."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_type=config["model"]["type"],
        metrics=metrics or {},
        features_count=model.n_features_in_,
    )


@app.post("/predict", response_model=RentOutput)
async def predict(input_data: RentInput):
    """Predict rent from input features."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build DataFrame with correct column names (matching training)
        data = {
            "BHK": [input_data.BHK],
            "Size": [input_data.Size],
            "Bathroom": [input_data.Bathroom],
            "floor_num": [input_data.floor_num],
            "total_floors": [input_data.total_floors],
            "Area Type": [input_data.Area_Type],
            "City": [input_data.City],
            "Furnishing Status": [input_data.Furnishing_Status],
            "Tenant Preferred": [input_data.Tenant_Preferred],
            "Area Locality": [input_data.Area_Locality],
        }
        df = pd.DataFrame(data)

        # Apply target encoding for high-cardinality features
        if target_encoding_maps:
            for col, encoding_map in target_encoding_maps.items():
                if col in df.columns:
                    global_mean = np.mean(list(encoding_map.values()))
                    df[col] = df[col].map(encoding_map).fillna(global_mean)

        # Transform using preprocessor
        X = preprocessor.transform(df)

        # Predict
        prediction = model.predict(X)[0]

        return RentOutput(
            predicted_rent=round(float(max(prediction, 0)), 2),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
