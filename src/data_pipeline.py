"""
Data pipeline — fetch raw data from S3, clean, validate, build features, upload.

Usage:
    python -m src.data_pipeline
    python -m src.data_pipeline --skip-download
"""
import os
import sys
import logging
import argparse

import pandas as pd
import numpy as np

from src.core_utils import (
    load_config, download_from_s3, upload_to_s3, upload_directory_to_s3,
    build_features, save_model,
)
from src.features import engineer_features
from src.locality_embeddings import generate_localities_json, add_city_coordinates, generate_locality_embeddings, apply_locality_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Cleaning helpers ─────────────────────────────────────────────────

FLOOR_LABELS = {"ground": 0, "upper basement": -1, "lower basement": -2}


def parse_floor(floor_str):
    """Parse 'X out of Y' → (floor_num, total_floors).

    Handles Ground, Upper/Lower Basement, numeric floors, and bad input (→ NaN).
    """
    try:
        if pd.isna(floor_str):
            return np.nan, np.nan

        parts = str(floor_str).split(" out of ")
        if len(parts) != 2:
            return np.nan, np.nan

        floor_part, total_part = parts[0].strip(), parts[1].strip()

        try:
            total_floors = int(total_part)
        except ValueError:
            total_floors = np.nan

        label = floor_part.lower()
        if label in FLOOR_LABELS:
            floor_num = FLOOR_LABELS[label]
        else:
            try:
                floor_num = int(floor_part)
            except ValueError:
                floor_num = np.nan

        return floor_num, total_floors
    except Exception:
        return np.nan, np.nan


def clean_data(df):
    """Drop junk columns, parse Floor, normalize locality, drop null rows."""
    logger.info(f"Raw data shape: {df.shape}")

    # Drop columns we don't need
    drop_cols = [c for c in ["Point of Contact", "Posted On"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    logger.info(f"Dropped columns: {drop_cols}")

    # Parse Floor → floor_num + total_floors
    if "Floor" in df.columns:
        parsed = df["Floor"].apply(parse_floor)
        df["floor_num"] = parsed.apply(lambda x: x[0])
        df["total_floors"] = parsed.apply(lambda x: x[1])
        df = df.drop(columns=["Floor"])
        logger.info("Parsed 'Floor' → 'floor_num', 'total_floors'")

    # Lowercase locality for consistent target encoding
    if "Area Locality" in df.columns:
        df["Area Locality"] = df["Area Locality"].str.lower()

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} null rows. Remaining: {len(df)}")

    return df


def cap_outliers(df, config):
    """Cap extreme rent values using the configured percentile threshold."""
    cap_pct = config["features"].get("outlier_cap_percentile", 99)
    target = config["features"]["target"]
    cap_val = df[target].quantile(cap_pct / 100)
    before_max = df[target].max()
    df[target] = df[target].clip(upper=cap_val)
    capped = (df[target] == cap_val).sum()
    logger.info(f"Capped {target} at {cap_pct}th percentile ({cap_val:.0f}). Max was {before_max:.0f}, {capped} rows capped.")
    return df


def validate_data(df, config):
    """Sanity-check cleaned data: no nulls, all expected columns present."""
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        raise ValueError(f"Cleaned data still has nulls:\n{nulls[nulls > 0]}")

    features = config["features"]
    target = features["target"]

    expected = features["numerical"] + features["categorical"] + features["high_cardinality"] + [target]
    
    emb_config = config.get("location", {}).get("embedding", {})
    if not emb_config.get("enabled", False):
        expected = [c for c in expected if not c.startswith("locality_emb_") and c not in ["city_lat", "city_lon"]]
    
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    logger.info(f"Validation passed — {df.shape[0]} rows, {df.shape[1]} cols")


# ── Main pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data pipeline: fetch → clean → features → S3")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-download", action="store_true", help="Skip S3 download if local file exists")
    args = parser.parse_args()

    config = load_config(args.config)
    s3 = config["s3"]
    data = config["data"]

    # 1. Get raw data
    raw_path = data["raw_path"]
    if not args.skip_download or not os.path.exists(raw_path):
        logger.info("Fetching raw data from S3...")
        ok = download_from_s3(s3["bucket"], s3["raw_key"], raw_path)
        if not ok:
            if os.path.exists(raw_path):
                logger.info("S3 download failed but local data exists, continuing...")
            else:
                logger.error("No data available — S3 download failed and no local file.")
                sys.exit(1)
    else:
        logger.info("Raw data already exists locally, skipping download")

    # 2. Clean
    logger.info("Cleaning data...")
    df_clean = clean_data(pd.read_csv(raw_path))

    # 2b. Cap outliers
    logger.info("Capping outliers...")
    df_clean = cap_outliers(df_clean, config)

    # 2c. Engineer features
    logger.info("Engineering features...")
    df_clean = engineer_features(df_clean)

    # 2d. Add city coordinates
    logger.info("Adding city coordinates...")
    df_clean = add_city_coordinates(df_clean)

    # 2e. Generate and apply locality embeddings
    loc_config = config.get("location", {})
    emb_config = loc_config.get("embedding", {})
    if emb_config.get("enabled", False):
        logger.info("Generating locality embeddings...")
        embeddings_map, pca = generate_locality_embeddings(df_clean, config)
        dim = emb_config.get("dimensions", 16)
        df_clean = apply_locality_embeddings(df_clean, embeddings_map, dim)
    else:
        logger.info("Locality embeddings disabled in config")

    # 3. Validate
    validate_data(df_clean, config)

    # 4. Save cleaned data
    processed_dir = data["processed_path"]
    os.makedirs(processed_dir, exist_ok=True)
    clean_path = os.path.join(processed_dir, "cleaned_data.csv")
    df_clean.to_csv(clean_path, index=False)
    logger.info(f"Saved cleaned data to {clean_path}")

    # 5. Build features (train/test split + preprocessing)
    logger.info("Building features...")
    X_train, X_test, y_train, y_test, preprocessor, encoding_maps = build_features(df_clean, config)

    # 6. Save preprocessing artifacts
    features_dir = os.path.join(processed_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    save_model(preprocessor, os.path.join(features_dir, "preprocessor.joblib"))
    save_model(encoding_maps, os.path.join(features_dir, "target_encoding_maps.joblib"))
    logger.info(f"Saved preprocessing artifacts to {features_dir}")

    # 7. Upload everything to S3
    logger.info("Uploading to S3...")
    ok1 = upload_to_s3(clean_path, s3["bucket"], s3["processed_key"])
    ok2 = upload_directory_to_s3(features_dir, s3["bucket"], s3["features_prefix"])

    # 8. Generate and upload localities JSON for frontend
    logger.info("Generating localities JSON...")
    localities_path = "data/processed/localities_by_city.json"
    generate_localities_json(df_clean, localities_path)
    ok3 = upload_to_s3(localities_path, s3["bucket"], s3.get("localities_key", "processed_data/localities_by_city.json"))

    if ok1 and ok2 and ok3:
        logger.info("Data pipeline complete — all files uploaded to S3!")
    else:
        logger.error("Some S3 uploads failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()