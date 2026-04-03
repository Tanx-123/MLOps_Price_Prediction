"""Shared feature engineering logic used by both training and serving."""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from existing columns.
    
    Must match the logic used during model training.
    """
    df = df.copy()
    df["size_per_bhk"] = df["Size"] / df["BHK"].clip(lower=1)
    df["bath_to_bhk_ratio"] = df["Bathroom"] / df["BHK"].clip(lower=1)
    df["floor_ratio"] = df["floor_num"] / df["total_floors"].clip(lower=1)
    return df
