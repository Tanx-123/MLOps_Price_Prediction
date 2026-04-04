"""
FastAPI app for serving rent predictions.

Features:
- Health checks with S3/model status
- Request logging for predictions
- Prometheus metrics

Endpoints:
    GET  /health       — Detailed health check
    GET  /metrics      — Prometheus metrics
    GET  /model/info   — Model metadata and metrics
    POST /predict      — Predict rent from features

Usage:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""
import os
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from src.core_utils import load_config, ensure_local_file, get_s3_client
from src.features import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Prometheus Metrics ──────────────────────────────────────────────────

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction duration')
MODEL_LOAD_STATUS = Gauge('model_loaded', 'Model loaded status (1=loaded, 0=not loaded)')
S3_AVAILABLE = Gauge('s3_available', 'S3 connectivity status (1=available, 0=not available)')


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
    status: str
    model_loaded: bool
    s3_available: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    metrics: dict
    features_count: int
    target_transform: Optional[str]


# ── Global state (populated on startup) ──────────────────────────────

model = None
preprocessor = None
target_encoding_maps = None
metrics = None
config = None
target_transform = None
s3_client = None
localities_data = None
locality_embeddings = None


# ── Logging setup ────────────────────────────────────────────────────

class PredictionLogger:
    """Log predictions to file for analysis."""
    
    def __init__(self, log_file: str = "logs/predictions.log"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, input_data: dict, prediction: float, latency_ms: float):
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": input_data,
            "prediction": prediction,
            "latency_ms": round(latency_ms, 2),
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


prediction_logger = PredictionLogger()


# ── Helper functions ─────────────────────────────────────────────────

def _check_s3_health() -> bool:
    """Check if S3 is accessible."""
    try:
        s3 = get_s3_client()
        if s3 is None:
            return False
        s3.head_bucket(Bucket=config["s3"]["bucket"])
        return True
    except Exception:
        return False


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
    global model, preprocessor, target_encoding_maps, metrics, config, target_transform, s3_client, localities_data, locality_embeddings

    config = load_config()
    s3_cfg = config["s3"]
    artifacts_dir = config["data"]["artifacts_path"]
    data_dir = config["data"]["processed_path"]
    bucket, prefix = s3_cfg["bucket"], s3_cfg["artifacts_prefix"]

    # Load localities data for frontend dropdown
    localities_path = os.path.join(data_dir, "localities_by_city.json")
    localities_s3_key = s3_cfg.get("localities_key", "processed_data/localities_by_city.json")
    if not os.path.exists(localities_path):
        if ensure_local_file(localities_path, bucket, localities_s3_key):
            logger.info("Downloaded localities file from S3")
    if os.path.exists(localities_path):
        with open(localities_path, "r") as f:
            localities_data = json.load(f)
        logger.info(f"Loaded localities for {len(localities_data)} cities")
    else:
        logger.warning(f"Localities file not found: {localities_path}")
        localities_data = {}

    # Load locality embeddings (only if enabled in config)
    emb_config = config.get("location", {}).get("embedding", {})
    locality_embeddings = None
    if emb_config.get("enabled", False):
        emb_path = os.path.join(artifacts_dir, "locality_embeddings.joblib")
        if os.path.exists(emb_path):
            locality_embeddings = joblib.load(emb_path)
            logger.info(f"Loaded locality embeddings with {len(locality_embeddings['embeddings_map'])} localities")
        else:
            emb_s3_key = emb_config.get("cache_path", "artifacts/locality_embeddings.joblib")
            if ensure_local_file(emb_path, bucket, emb_s3_key):
                locality_embeddings = joblib.load(emb_path)
                logger.info(f"Loaded locality embeddings with {len(locality_embeddings['embeddings_map'])} localities")
            else:
                logger.warning("Locality embeddings not found")
    else:
        logger.info("Locality embeddings disabled in config")

    # Check S3 connectivity
    s3_client = _check_s3_health()
    S3_AVAILABLE.set(1 if s3_client else 0)

    # Load artifacts
    model = _load_joblib(artifacts_dir, bucket, prefix, "best_model.joblib")
    preprocessor = _load_joblib(artifacts_dir, bucket, prefix, "preprocessor.joblib")
    target_encoding_maps = _load_joblib(artifacts_dir, bucket, prefix, "target_encoding_maps.joblib")
    metrics = _load_json(artifacts_dir, "ensemble_results.json", "metrics.json")
    target_transform = config.get("features", {}).get("target_transform", None)

    MODEL_LOAD_STATUS.set(1)
    logger.info(f"Loaded model: {model.__class__.__name__}")
    logger.info(f"S3 available: {s3_client}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield
    # Cleanup on shutdown
    logger.info("Shutting down...")


# ── App setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="Rent Predictor API",
    description="ML-powered rent prediction",
    version="1.0.1",
    lifespan=lifespan,
)

# CORS: allow specific origins from env var, default to all for dev
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request timing to all requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)
    
    return response


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
    s3_ok = _check_s3_health()
    S3_AVAILABLE.set(1 if s3_ok else 0)
    
    # Health depends on model being loaded; S3 is optional if model is cached
    if model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            s3_available=s3_ok,
            version="1.0.1",
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        s3_available=s3_ok,
        version="1.0.1",
    )


@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    return {"content": generate_latest().decode("utf-8")}


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return ModelInfoResponse(
        model_type=config["model"]["type"],
        metrics=metrics or {},
        features_count=getattr(model, "n_features_in_", 0),
        target_transform=target_transform,
    )


@app.get("/localities")
async def get_localities():
    """Return cities and their localities for frontend dropdown."""
    if not localities_data:
        raise HTTPException(404, "Localities data not available")
    return localities_data


@app.post("/predict", response_model=RentOutput)
async def predict(input_data: RentInput):
    if model is None or preprocessor is None:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status=503).inc()
        raise HTTPException(503, "Model not loaded")

    start_time = time.time()
    
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

        # Engineer features (same logic as training pipeline)
        df = engineer_features(df)

        # Add city coordinates
        cities = config.get("location", {}).get("cities", {
            "mumbai": (19.0760, 72.8777),
            "bangalore": (12.9716, 77.5946),
            "chennai": (13.0827, 80.2707),
            "hyderabad": (17.3850, 78.4867),
            "delhi": (28.7041, 77.1025),
            "kolkata": (22.5726, 88.3639),
        })
        city_key = input_data.City.lower() if input_data.City else "bangalore"
        coords = cities.get(city_key, (0.0, 0.0))
        df["city_lat"] = float(coords[0])
        df["city_lon"] = float(coords[1])

        # Apply locality embeddings
        if locality_embeddings:
            emb_map = locality_embeddings["embeddings_map"]
            dim = locality_embeddings.get("dimensions", 16)
            city_loc = input_data.Area_Locality.lower() + ", " + input_data.City.lower()
            emb = emb_map.get(city_loc, np.zeros(dim))
            for i in range(dim):
                df[f"locality_emb_{i}"] = emb[i]

        # Apply target encoding (same logic as training)
        if target_encoding_maps:
            for col, enc_map in target_encoding_maps.items():
                if col in df.columns:
                    if isinstance(enc_map, dict) and "map" in enc_map:
                        mapping = enc_map.get("map", {})
                        fallback = enc_map.get("global_mean", 0.0)
                    else:
                        mapping = enc_map if isinstance(enc_map, dict) else {}
                        fallback = np.mean(list(mapping.values())) if mapping else 0.0

                    df[col] = df[col].map(mapping).fillna(fallback)

        X = preprocessor.transform(df)
        pred = float(model.predict(X)[0])
        if target_transform == "log1p":
            pred = np.expm1(pred)
        
        pred = round(max(pred, 0), 2)
        
        # Log prediction
        latency_ms = (time.time() - start_time) * 1000
        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(latency_ms / 1000)
        
        prediction_logger.log(
            input_data=input_data.model_dump(),
            prediction=pred,
            latency_ms=latency_ms,
        )
        
        logger.info(f"Prediction: Rs.{pred:,} (latency: {latency_ms:.2f}ms)")
        
        return RentOutput(predicted_rent=pred)

    except Exception:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status=500).inc()
        logger.exception("Prediction failed due to server error")
        raise HTTPException(500, "Prediction failed due to server error")
