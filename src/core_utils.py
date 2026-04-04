"""
Core utilities — S3 helpers, config, model I/O, preprocessing, and metrics.
"""
import os
import logging
import json
import pathlib
import time
from functools import wraps

import yaml
import pandas as pd
import numpy as np
import joblib
import boto3
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features import engineer_features
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator that retries a function on exception with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = delay * (backoff ** attempt)
                    logger.warning(f"{func.__name__} failed: {e}. Retrying in {wait:.1f}s...")
                    time.sleep(wait)
        return wrapper
    return decorator


# ── Environment & Config ─────────────────────────────────────────────

def load_env():
    """Find and load .env from cwd or up to two parents."""
    for path in [pathlib.Path.cwd(), *list(pathlib.Path.cwd().parents)[:2]]:
        env_path = path / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded .env from {env_path}")
            return
    logger.warning("No .env file found, using system environment variables")


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── S3 Operations ────────────────────────────────────────────────────

def get_s3_client(region=None):
    """Build an S3 client from env-var credentials. Returns None if missing."""
    load_env()
    key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    if not key_id or not secret:
        logger.error("AWS credentials not found in environment variables")
        return None

    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )


@with_retry(max_attempts=3, delay=1.0, backoff=2.0)
def upload_to_s3(local_path, bucket, key, region=None):
    """Upload a local file to S3. Returns True on success."""
    if not os.path.exists(local_path):
        logger.error(f"File not found: {local_path}")
        return False

    s3 = get_s3_client(region)
    if s3 is None:
        return False

    try:
        logger.info(f"Uploading {local_path} → s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)
        logger.info("Upload successful!")
        return True
    except ClientError as e:
        logger.error(f"S3 upload error: {e}")
        return False
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False


@with_retry(max_attempts=3, delay=1.0, backoff=2.0)
def download_from_s3(bucket, key, local_path, region=None):
    """Download a file from S3. Returns True on success."""
    s3 = get_s3_client(region)
    if s3 is None:
        return False

    # Check existence first
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            logger.error(f"S3 object not found: s3://{bucket}/{key}")
        else:
            logger.error(f"Error checking S3 object: {e}")
        return False

    try:
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        logger.info(f"Downloading s3://{bucket}/{key} → {local_path}")
        s3.download_file(bucket, key, local_path)
        logger.info(f"Downloaded to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def upload_directory_to_s3(local_dir, bucket, s3_prefix, region=None):
    """Upload every file in a directory to S3 under a prefix."""
    success = True
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_prefix}/{rel}".replace("\\", "/")
            if not upload_to_s3(local_path, bucket, s3_key, region):
                success = False
    return success


def ensure_local_file(local_path, bucket, s3_key):
    """Return True if file exists locally; if not, try downloading from S3."""
    if os.path.exists(local_path):
        logger.info(f"Found locally: {local_path}")
        return True
    logger.info(f"Not found locally, downloading from S3: {s3_key}")
    return download_from_s3(bucket, s3_key, local_path)


# ── File I/O ─────────────────────────────────────────────────────────

def save_json(data, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_model(obj, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path):
    return joblib.load(path)


# ── Target Encoding ──────────────────────────────────────────────────

def target_encode(df, column, target, smoothing=10):
    """Smoothed target encoding for a high-cardinality column.

    Returns (encoded_series, encoding_artifact_dict).

    `encoding_artifact_dict` contains:
      - `map`: category -> encoded value
      - `global_mean`: fallback used for unseen categories
    """
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(["mean", "count"])
    smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)

    category_map = smooth.to_dict()
    encoded = df[column].map(category_map).fillna(global_mean)
    return encoded, {"map": category_map, "global_mean": float(global_mean)}


def apply_target_encoding(df, column, encoding_artifact):
    """Apply a pre-computed target encoding artifact to a column."""
    mapping = encoding_artifact.get("map", {}) if isinstance(encoding_artifact, dict) else {}
    global_mean = encoding_artifact.get("global_mean", 0.0) if isinstance(encoding_artifact, dict) else 0.0
    return df[column].map(mapping).fillna(global_mean)


class TargetEncodingTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible supervised target encoder.

    Fits a smoothed target-encoding map per column using (X, y), then transforms
    those columns into numeric encoded values.
    """

    def __init__(self, columns, smoothing=10):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X, y):
        # Expect X as a pandas.DataFrame (we index by column names).
        if not hasattr(X, "columns"):
            raise TypeError("TargetEncodingTransformer expects X as a pandas.DataFrame")

        self.global_mean_ = float(np.mean(y))
        self.encoding_maps_ = {}

        y_arr = np.asarray(y)
        for col in self.columns:
            if col not in X.columns:
                continue

            agg_df = pd.DataFrame({col: X[col], "_target": y_arr}).groupby(col)["_target"].agg(["mean", "count"])
            smooth = (agg_df["count"] * agg_df["mean"] + self.smoothing * self.global_mean_) / (
                agg_df["count"] + self.smoothing
            )
            self.encoding_maps_[col] = {
                "map": smooth.to_dict(),
                "global_mean": self.global_mean_,
            }

        return self

    def transform(self, X):
        if not hasattr(X, "columns"):
            raise TypeError("TargetEncodingTransformer expects X as a pandas.DataFrame")

        X_out = X.copy()
        for col in self.columns:
            if col not in X_out.columns:
                continue
            if col not in self.encoding_maps_:
                # Column existed in data but wasn't seen during fit.
                X_out[col] = pd.Series(np.full(len(X_out), self.global_mean_), index=X_out.index)
                continue

            encoding_artifact = self.encoding_maps_[col]
            X_out[col] = X_out[col].map(encoding_artifact["map"]).fillna(encoding_artifact["global_mean"])

        return X_out


# ── Preprocessing ────────────────────────────────────────────────────

def build_features(df, config):
    """Split data, apply target encoding + sklearn transforms.

    Returns (X_train, X_test, y_train, y_test, preprocessor, encoding_maps).
    Target encoding is fit on train only to prevent data leakage.
    If target_transform is 'log1p', y values are log-transformed.
    """
    features = config["features"]
    target_col = features["target"]
    test_size = config["model"].get("test_size", 0.2)
    random_state = config["model"].get("random_state", 42)
    smoothing = features.get("target_encoding_smoothing", 10)
    target_transform = features.get("target_transform", None)

    df = engineer_features(df)

    loc_config = config.get("location", {})
    emb_config = loc_config.get("embedding", {})

    if emb_config.get("enabled", False):
        from src.locality_embeddings import add_city_coordinates, apply_locality_embeddings
        df = add_city_coordinates(df, config)
        embeddings_map, pca = None, None
        emb_path = emb_config.get("cache_path", "artifacts/locality_embeddings.joblib")
        if os.path.exists(emb_path):
            cached = joblib.load(emb_path)
            dim = emb_config.get("dimensions", 16)
            df = apply_locality_embeddings(df, cached["embeddings_map"], dim)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # Target-encode high-cardinality cols (fit on train, apply to test)
    encoding_maps = {}
    for col in features.get("high_cardinality", []):
        train_df[col], enc_artifact = target_encode(train_df, col, target_col, smoothing=smoothing)
        encoding_maps[col] = enc_artifact
        test_df[col] = apply_target_encoding(test_df, col, enc_artifact)

    # Build feature lists dynamically based on what columns actually exist
    num_cols = [c for c in features["numerical"] + features.get("high_cardinality", []) if c in train_df.columns]
    cat_cols = [c for c in features["categorical"] if c in train_df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Apply log transform to target if configured
    if target_transform == "log1p":
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
        logger.info("Applied log1p transform to target variable")

    logger.info(f"Feature matrix — train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor, encoding_maps


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }