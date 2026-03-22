"""
Core utilities — S3 helpers, config, model I/O, preprocessing, and metrics.
"""
import os
import logging
import json
import pathlib

import yaml
import pandas as pd
import numpy as np
import joblib
import boto3
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


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

    Returns (encoded_series, encoding_map_dict).
    """
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(["mean", "count"])
    smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)

    encoding_map = smooth.to_dict()
    encoded = df[column].map(encoding_map).fillna(global_mean)
    return encoded, encoding_map


def apply_target_encoding(df, column, encoding_map, global_mean):
    """Apply a pre-computed target encoding map to a column."""
    return df[column].map(encoding_map).fillna(global_mean)


# ── Preprocessing ────────────────────────────────────────────────────

def build_features(df, config):
    """Split data, apply target encoding + sklearn transforms.

    Returns (X_train, X_test, y_train, y_test, preprocessor, encoding_maps).
    Target encoding is fit on train only to prevent data leakage.
    """
    features = config["features"]
    target_col = features["target"]
    test_size = config["model"].get("test_size", 0.2)
    random_state = config["model"].get("random_state", 42)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # Target-encode high-cardinality cols (fit on train, apply to test)
    encoding_maps = {}
    for col in features.get("high_cardinality", []):
        train_df[col], enc_map = target_encode(train_df, col, target_col)
        encoding_maps[col] = enc_map
        global_mean = train_df[target_col].mean()
        test_df[col] = apply_target_encoding(test_df, col, enc_map, global_mean)

    num_cols = features["numerical"] + features.get("high_cardinality", [])
    cat_cols = features["categorical"]

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

    logger.info(f"Feature matrix — train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor, encoding_maps


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }