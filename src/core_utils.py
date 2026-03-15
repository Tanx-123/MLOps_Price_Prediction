"""
Core utilities for the ML pipeline.
Consolidated from utils.py and s3_utils.py to eliminate redundancy.
"""

import os
import logging
import json
import yaml
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import joblib
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def get_s3_client(region: Optional[str] = None):
    """Create and return an S3 client using credentials from environment variables."""
    # Load .env file first (for local development)
    load_dotenv()
    
    # Get credentials from environment variables (works in both local and CI/CD)
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    if not aws_access_key or not aws_secret_key:
        logger.error("AWS credentials not found in environment variables")
        return None
        
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
    )


def s3_object_exists(bucket: str, key: str, region: Optional[str] = None) -> bool:
    """Check if an S3 object exists."""
    s3 = get_s3_client(region)
    if s3 is None:
        return False
    
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        logger.error(f"Error checking S3 object: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking S3 object: {e}")
        return False


def upload_to_s3(local_path: str, bucket: str, key: str, region: Optional[str] = None) -> bool:
    """Upload a local file to S3."""
    if not os.path.exists(local_path):
        logger.error(f"File not found: {local_path}")
        return False

    try:
        s3 = get_s3_client(region)
        logger.info(f"Uploading {local_path} → s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)
        logger.info("Upload successful!")
        return True

    except NoCredentialsError:
        logger.error("AWS credentials missing. Check .env file.")
        return False
    except PartialCredentialsError:
        logger.error("Incomplete AWS credentials in .env file.")
        return False
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "AccessDenied":
            logger.error(f"Permission denied for bucket {bucket}")
        elif code == "NoSuchBucket":
            logger.error(f"Bucket {bucket} does not exist")
        else:
            logger.error(f"S3 error: {e}")
        return False
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False


def download_from_s3(bucket: str, key: str, local_path: str, region: Optional[str] = None) -> bool:
    """Download a file from S3 to local path with improved error handling."""
    # Check if object exists first
    if not s3_object_exists(bucket, key, region):
        logger.error(f"S3 object not found: s3://{bucket}/{key}")
        return False
    
    try:
        s3 = get_s3_client(region)
        if s3 is None:
            return False
            
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logger.info(f"Downloading s3://{bucket}/{key} → {local_path}")
        s3.download_file(bucket, key, local_path)
        logger.info(f"Downloaded to {local_path}")
        return True

    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchBucket":
            logger.error(f"Bucket {bucket} not found")
        elif code == "AccessDenied":
            logger.error(f"Permission denied for s3://{bucket}/{key}")
        else:
            logger.error(f"S3 error: {e}")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def upload_directory_to_s3(local_dir: str, bucket: str, s3_prefix: str, region: Optional[str] = None) -> bool:
    """Upload all files in a local directory to S3 under a prefix."""
    success = True
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")
            if not upload_to_s3(local_path, bucket, s3_key, region):
                success = False
    return success


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_directory(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save data to JSON file."""
    ensure_directory(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_model(model: Any, path: str) -> None:
    """Save model to joblib file."""
    ensure_directory(path)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Load model from joblib file."""
    return joblib.load(path)


def target_encode(df: pd.DataFrame, column: str, target: str, smoothing: int = 10) -> Tuple[pd.Series, Dict]:
    """Apply target encoding to a high-cardinality categorical column."""
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(["mean", "count"])
    smooth_mean = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
        agg["count"] + smoothing
    )
    encoding_map = smooth_mean.to_dict()
    encoded = df[column].map(encoding_map).fillna(global_mean)
    return encoded, encoding_map


def target_encode_with_map(df: pd.DataFrame, column: str, encoding_map: Dict, global_mean: float) -> pd.Series:
    """Apply target encoding using a pre-computed map."""
    return df[column].map(encoding_map).fillna(global_mean)


def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple:
    """Preprocess data with proper train/test split to prevent data leakage."""
    features = config["features"]
    target_col = features["target"]
    model_config = config["model"]

    test_size = model_config.get("test_size", 0.2)
    random_state = model_config.get("random_state", 42)
    
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=None
    )
    
    logger.info(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

    target_encoding_maps = {}
    for col in features.get("high_cardinality", []):
        logger.info(f"Target-encoding column: {col}")
        train_df[col], encoding_map = target_encode(train_df, col, target_col)
        target_encoding_maps[col] = encoding_map
        
        # Apply the same encoding to test set (using training data's encoding)
        global_mean = train_df[target_col].mean()
        test_df[col] = target_encode_with_map(test_df, col, encoding_map, global_mean)

    numerical_cols = features["numerical"] + features.get("high_cardinality", [])
    categorical_cols = features["categorical"]

    logger.info(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
    logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    logger.info(f"Train feature matrix shape: {X_train.shape}")
    logger.info(f"Test feature matrix shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, preprocessor, target_encoding_maps


def compute_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Compute regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(mean_squared_error(y_true, y_pred, squared=False)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def ensure_local_file(local_path: str, bucket: str, s3_key: str) -> bool:
    """Ensure a file exists locally; download from S3 if missing."""
    if os.path.exists(local_path):
        logger.info(f"Found locally: {local_path}")
        return True
    logger.info(f"Not found locally, downloading from S3: {s3_key}")
    return download_from_s3(bucket, s3_key, local_path)
