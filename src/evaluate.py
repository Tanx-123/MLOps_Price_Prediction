"""
Evaluate the trained model using the new pipeline. Fetch artifacts from S3 if not present locally.

Pipeline:
    1. Download cleaned data from S3
    2. Preprocess with saved preprocessing artifacts
    3. Split into train/test with same random state
    4. Evaluate model on test set
    5. Compute MAE, RMSE, R²
    6. Save metrics locally + upload to S3
    7. Exit code 1 if R² < threshold (quality gate)

Usage:
    python -m src.evaluate
"""

import os
import json
import logging
import argparse

import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.s3_utils import download_from_s3, upload_to_s3
from src.utils import load_config, preprocess_for_evaluation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ensure_local_file(local_path, bucket, s3_key):
    """Ensure a file exists locally; download from S3 if missing."""
    if os.path.exists(local_path):
        logger.info(f"Found locally: {local_path}")
        return True
    logger.info(f"Not found locally, downloading from S3: {s3_key}")
    return download_from_s3(bucket, s3_key, local_path)


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate rent prediction model")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()
    config = load_config(args.config)
    s3_config = config["s3"]
    data_config = config["data"]
    model_config = config["model"]

    artifacts_dir = data_config["artifacts_path"]
    processed_dir = data_config["processed_path"]
    bucket = s3_config["bucket"]
    # Step 1: Ensure all essential artifacts are available
    logger.info("Step 1: Loading production artifacts...")    
    # Load best model
    model_path = os.path.join(artifacts_dir, "best_model.joblib")
    if not ensure_local_file(model_path, bucket, f"{s3_config['artifacts_prefix']}/best_model.joblib"):
        logger.error("Failed to load best model.")
        exit(1)
    model = joblib.load(model_path)
    logger.info(f"Loaded best model: {model.__class__.__name__}")

    # Load preprocessing artifacts
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    if not ensure_local_file(preprocessor_path, bucket, f"{s3_config['artifacts_prefix']}/preprocessor.joblib"):
        logger.error("Failed to load preprocessor.")
        exit(1)
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded preprocessor")

    target_encoding_maps_path = os.path.join(artifacts_dir, "target_encoding_maps.joblib")
    if not ensure_local_file(target_encoding_maps_path, bucket, f"{s3_config['artifacts_prefix']}/target_encoding_maps.joblib"):
        logger.error("Failed to load target encoding maps.")
        exit(1)
    target_encoding_maps = joblib.load(target_encoding_maps_path)
    logger.info("Loaded target encoding maps")

    # Step 2: Download cleaned data for evaluation
    logger.info("Step 2: Downloading cleaned data for evaluation...")
    clean_path = os.path.join(processed_dir, "cleaned_data.csv")
    if not ensure_local_file(clean_path, bucket, s3_config["processed_key"]):
        logger.error("Failed to load cleaned data.")
        exit(1)

    # Step 3: Preprocess data for evaluation
    logger.info("Step 3: Preprocessing data for evaluation...")
    df = pd.read_csv(clean_path)
    X_test, y_test = preprocess_for_evaluation(df, config, preprocessor, target_encoding_maps)

    # Step 4: Predict and evaluate
    logger.info("Step 4: Evaluating model...")
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best Model: {model.__class__.__name__}")
    logger.info(f"Test Set Size: {len(y_test)}")
    logger.info("-" * 60)
    logger.info(f"  MAE  : {metrics['mae']:>12,.2f}")
    logger.info(f"  RMSE : {metrics['rmse']:>12,.2f}")
    logger.info(f"  R²   : {metrics['r2']:>12.4f}")
    logger.info("=" * 60)

    # Step 5: Save metrics
    os.makedirs(artifacts_dir, exist_ok=True)
    eval_metrics_path = os.path.join(artifacts_dir, "eval_metrics.json")
    eval_metrics = {
        "evaluation": metrics,
        "model_type": model.__class__.__name__,
        "test_size": len(y_test),
        "evaluation_date": pd.Timestamp.now().isoformat()
    }
    
    with open(eval_metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {eval_metrics_path}")

    # Upload metrics to S3
    upload_to_s3(
        eval_metrics_path,
        bucket,
        f"{s3_config['artifacts_prefix']}/eval_metrics.json",
    )

    # Step 6: Quality gate
    r2_threshold = model_config.get("r2_threshold", 0.75)  # Increased threshold
    if metrics["r2"] < r2_threshold:
        logger.error(
            f"QUALITY GATE FAILED: R² ({metrics['r2']}) < threshold ({r2_threshold})"
        )
        logger.error("Model performance is below acceptable threshold.")
        exit(1)
    else:
        logger.info(
            f"QUALITY GATE PASSED: R² ({metrics['r2']}) >= threshold ({r2_threshold})"
        )
        logger.info("Model performance meets quality standards.")

    logger.info("Evaluation pipeline complete!")
    logger.info(f"Model ready for production serving.")


if __name__ == "__main__":
    main()
