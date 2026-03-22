"""
Evaluate the trained model on the held-out test split.

Exits with code 1 if R² < threshold (quality gate).

Usage:
    python -m src.evaluate
"""
import os
import sys
import json
import logging
import argparse

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src.core_utils import load_config, ensure_local_file, upload_to_s3, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def apply_target_encoding(df, encoding_maps):
    """Apply pre-computed target encoding maps to the dataframe."""
    for col, enc_map in encoding_maps.items():
        if col in df.columns:
            fallback = np.mean(list(enc_map.values())) if enc_map else 0
            df[col] = df[col].map(enc_map).fillna(fallback)
    return df


def load_artifact(path, bucket, prefix, filename):
    """Load a joblib artifact, downloading from S3 if needed."""
    full_path = os.path.join(path, filename)
    if not ensure_local_file(full_path, bucket, f"{prefix}/{filename}"):
        logger.error(f"Failed to load {filename}")
        sys.exit(1)
    return joblib.load(full_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    s3_cfg = config["s3"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    artifacts_dir = data_cfg["artifacts_path"]
    bucket = s3_cfg["bucket"]
    prefix = s3_cfg["artifacts_prefix"]

    # 1. Load artifacts
    logger.info("Loading model artifacts...")
    model = load_artifact(artifacts_dir, bucket, prefix, "best_model.joblib")
    preprocessor = load_artifact(artifacts_dir, bucket, prefix, "preprocessor.joblib")
    encoding_maps = load_artifact(artifacts_dir, bucket, prefix, "target_encoding_maps.joblib")
    logger.info(f"Loaded model: {model.__class__.__name__}")

    # 2. Load cleaned data
    clean_path = os.path.join(data_cfg["processed_path"], "cleaned_data.csv")
    if not ensure_local_file(clean_path, bucket, s3_cfg["processed_key"]):
        logger.error("Failed to load cleaned data.")
        sys.exit(1)

    # 3. Reproduce the same train/test split used during training
    df = pd.read_csv(clean_path)
    test_size = model_cfg.get("test_size", 0.2)
    random_state = model_cfg.get("random_state", 42)
    _, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    logger.info(f"Test set: {len(test_df)} rows (from {len(df)} total)")

    # 4. Preprocess test data and predict
    test_df = apply_target_encoding(test_df, encoding_maps)
    target_col = config["features"]["target"]
    y_test = test_df[target_col].values
    X_test = preprocessor.transform(test_df)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Model : {model.__class__.__name__}")
    logger.info(f"  MAE   : {metrics['mae']:>10,.2f}")
    logger.info(f"  RMSE  : {metrics['rmse']:>10,.2f}")
    logger.info(f"  R²    : {metrics['r2']:>10.4f}")
    logger.info("=" * 50)

    # 5. Save & upload metrics
    os.makedirs(artifacts_dir, exist_ok=True)
    eval_path = os.path.join(artifacts_dir, "eval_metrics.json")
    eval_data = {
        "evaluation": metrics,
        "model_type": model.__class__.__name__,
        "test_size": len(y_test),
        "evaluation_date": pd.Timestamp.now().isoformat(),
    }
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    upload_to_s3(eval_path, bucket, f"{prefix}/eval_metrics.json")

    # 6. Quality gate
    threshold = model_cfg.get("r2_threshold", 0.75)
    if metrics["r2"] < threshold:
        logger.error(f"QUALITY GATE FAILED: R² {metrics['r2']} < {threshold}")
        sys.exit(1)

    logger.info(f"QUALITY GATE PASSED: R² {metrics['r2']} >= {threshold}")
    logger.info("Model ready for production.")


if __name__ == "__main__":
    main()
