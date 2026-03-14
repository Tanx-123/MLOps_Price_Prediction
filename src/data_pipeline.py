"""
Unified data pipeline for the ML project.
Consolidated from fetch_data.py, data_cleaning.py, and feature_engineering.py.

Pipeline:
    1. Download raw data from S3 (if needed)
    2. Clean data (drop nulls, parse Floor, drop unnecessary columns)
    3. Validate cleaned data
    4. Save locally to data/processed/cleaned_data.csv
    5. Build features with proper train/test split
    6. Save preprocessing artifacts
    7. Upload to S3

Usage:
    python -m src.data_pipeline
"""

import os
import logging
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.core_utils import (
    load_config, download_from_s3, upload_to_s3, upload_directory_to_s3,
    target_encode, target_encode_with_map, ensure_directory, save_model
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_floor(floor_str):
    """Parse 'X out of Y' floor string into (floor_num, total_floors).

    Handles special cases:
        - 'Ground out of 2' → (0, 2)
        - 'Upper Basement out of 2' → (-1, 2)
        - 'Lower Basement out of 2' → (-2, 2)
        - '2 out of 5' → (2, 5)
    """
    try:
        if pd.isna(floor_str):
            return np.nan, np.nan

        parts = str(floor_str).split(" out of ")
        if len(parts) != 2:
            return np.nan, np.nan

        floor_part = parts[0].strip()
        total_part = parts[1].strip()

        # Parse total floors
        try:
            total_floors = int(total_part)
        except ValueError:
            total_floors = np.nan

        # Parse floor number
        floor_map = {
            "ground": 0,
            "upper basement": -1,
            "lower basement": -2,
        }
        floor_lower = floor_part.lower()
        if floor_lower in floor_map:
            floor_num = floor_map[floor_lower]
        else:
            try:
                floor_num = int(floor_part)
            except ValueError:
                floor_num = np.nan

        return floor_num, total_floors
    except Exception:
        return np.nan, np.nan


def clean_data(df):
    """Clean the raw rent dataset.

    Steps:
        1. Drop 'Point of Contact' and 'Posted On' columns
        2. Parse 'Floor' into 'floor_num' and 'total_floors'
        3. Drop original 'Floor' column
        4. Drop rows with any remaining null values

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    logger.info(f"Raw data shape: {df.shape}")

    # Drop unnecessary columns
    cols_to_drop = ["Point of Contact", "Posted On"]
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    logger.info(f"Dropped columns: {existing_drops}")

    # Parse Floor column
    if "Floor" in df.columns:
        floor_parsed = df["Floor"].apply(parse_floor)
        df["floor_num"] = floor_parsed.apply(lambda x: x[0])
        df["total_floors"] = floor_parsed.apply(lambda x: x[1])
        df = df.drop(columns=["Floor"])
        logger.info("Parsed 'Floor' → 'floor_num', 'total_floors'")

    # Drop rows with nulls
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    logger.info(f"Dropped {before - after} rows with nulls. Remaining: {after}")

    return df


def validate_data(df, config):
    """Validate cleaned data has expected columns and no nulls.

    Args:
        df: Cleaned DataFrame.
        config: Config dictionary.

    Raises:
        ValueError: If validation fails.
    """
    # Check no nulls remain
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Cleaned data still has nulls:\n{null_counts[null_counts > 0]}")

    # Check target column exists
    target = config["features"]["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in cleaned data")

    # Check expected feature columns exist
    expected_cols = (
        config["features"]["numerical"]
        + config["features"]["categorical"]
        + config["features"]["high_cardinality"]
        + [target]
    )
    # Some numerical features (floor_num, total_floors) are created during cleaning
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in cleaned data")

    logger.info(f"Data validation passed. Shape: {df.shape}, Columns: {list(df.columns)}")


def build_features(df, config):
    """Build features with proper train/test split to prevent data leakage."""
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


def main():
    parser = argparse.ArgumentParser(description="Complete data pipeline from raw data to features")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading raw data if local file exists")
    args = parser.parse_args()

    config = load_config(args.config)
    s3_config = config["s3"]
    data_config = config["data"]

    # Step 1: Download raw data from S3 if needed
    raw_path = data_config["raw_path"]
    if not args.skip_download or not os.path.exists(raw_path):
        logger.info("Step 1: Fetching raw data from S3...")
        success = download_from_s3(
            bucket=s3_config["bucket"],
            key=s3_config["raw_key"],
            local_path=raw_path,
        )
        if not success:
            logger.error("Failed to fetch raw data from S3. Exiting.")
            exit(1)
    else:
        logger.info("Step 1: Raw data already exists locally, skipping download")

    # Step 2: Load and clean
    logger.info("Step 2: Cleaning data...")
    df = pd.read_csv(raw_path)
    df_clean = clean_data(df)

    # Step 3: Validate
    logger.info("Step 3: Validating cleaned data...")
    validate_data(df_clean, config)

    # Step 4: Save locally
    processed_dir = data_config["processed_path"]
    os.makedirs(processed_dir, exist_ok=True)
    clean_path = os.path.join(processed_dir, "cleaned_data.csv")
    df_clean.to_csv(clean_path, index=False)
    logger.info(f"Step 4: Saved cleaned data to {clean_path}")
    # Step 5: Build features
    logger.info("Step 5: Building features with proper train/test split...")
    X_train, X_test, y_train, y_test, preprocessor, target_encoding_maps = build_features(df_clean, config)

    # Step 6: Save preprocessing artifacts (needed for serving)
    features_dir = os.path.join(processed_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    save_model(preprocessor, os.path.join(features_dir, "preprocessor.joblib"))
    save_model(target_encoding_maps, os.path.join(features_dir, "target_encoding_maps.joblib"))
    
    logger.info(f"Step 6: Saved preprocessing artifacts to {features_dir}")

    # Step 7: Upload to S3
    logger.info("Step 7: Uploading data and features to S3...")
    
    # Upload cleaned data
    success1 = upload_to_s3(
        clean_path,
        s3_config["bucket"],
        s3_config["processed_key"],
    )
    
    # Upload features
    success2 = upload_directory_to_s3(
        features_dir,
        s3_config["bucket"],
        s3_config["features_prefix"],
    )
    
    if success1 and success2:
        logger.info("Data pipeline complete! All files uploaded to S3 successfully!")
    else:
        logger.error("Some uploads failed.")
        exit(1)

    logger.info("Data pipeline complete!")

if __name__ == "__main__":
    main()