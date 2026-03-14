"""
Tests for data_pipeline.py.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline import clean_data, parse_floor, validate_data, load_config
from src.data_pipeline import target_encode, build_features


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Create a sample raw DataFrame matching the dataset schema."""
    return pd.DataFrame({
        "Posted On": ["2022-05-18", "2022-05-13", "2022-05-16", "2022-05-10"],
        "BHK": [2, 3, 2, 4],
        "Rent": [10000, 25000, 15000, 50000],
        "Size": [800, 1200, 900, 2000],
        "Floor": ["1 out of 3", "Ground out of 5", "2 out of 4", "Upper Basement out of 3"],
        "Area Type": ["Super Area", "Carpet Area", "Built Area", "Super Area"],
        "Area Locality": ["Whitefield", "Koramangala", "Whitefield", "Indiranagar"],
        "City": ["Bangalore", "Bangalore", "Chennai", "Bangalore"],
        "Furnishing Status": ["Furnished", "Semi-Furnished", "Unfurnished", "Furnished"],
        "Tenant Preferred": ["Bachelors/Family", "Family", "Bachelors", "Family"],
        "Bathroom": [2, 3, 2, 4],
        "Point of Contact": ["Contact Owner", "Contact Agent", "Contact Owner", "Contact Owner"],
    })


@pytest.fixture
def sample_cleaned_df():
    """Create a sample cleaned DataFrame (post data_cleaning)."""
    return pd.DataFrame({
        "BHK": [2, 3, 2, 4],
        "Rent": [10000, 25000, 15000, 50000],
        "Size": [800, 1200, 900, 2000],
        "Area Type": ["Super Area", "Carpet Area", "Built Area", "Super Area"],
        "Area Locality": ["Whitefield", "Koramangala", "Whitefield", "Indiranagar"],
        "City": ["Bangalore", "Bangalore", "Chennai", "Bangalore"],
        "Furnishing Status": ["Furnished", "Semi-Furnished", "Unfurnished", "Furnished"],
        "Tenant Preferred": ["Bachelors/Family", "Family", "Bachelors", "Family"],
        "Bathroom": [2, 3, 2, 4],
        "floor_num": [1, 0, 2, -1],
        "total_floors": [3, 5, 4, 3],
    })


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


# ── parse_floor tests ────────────────────────────────────────────────

class TestParseFloor:
    def test_normal_floor(self):
        assert parse_floor("2 out of 5") == (2, 5)

    def test_ground_floor(self):
        assert parse_floor("Ground out of 3") == (0, 3)

    def test_upper_basement(self):
        assert parse_floor("Upper Basement out of 2") == (-1, 2)

    def test_lower_basement(self):
        assert parse_floor("Lower Basement out of 4") == (-2, 4)

    def test_nan_input(self):
        result = parse_floor(np.nan)
        assert np.isnan(result[0]) and np.isnan(result[1])

    def test_invalid_format(self):
        result = parse_floor("invalid")
        assert np.isnan(result[0]) and np.isnan(result[1])


# ── clean_data tests ─────────────────────────────────────────────────

class TestCleanData:
    def test_drops_unnecessary_columns(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert "Point of Contact" not in result.columns
        assert "Posted On" not in result.columns

    def test_parses_floor(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert "floor_num" in result.columns
        assert "total_floors" in result.columns
        assert "Floor" not in result.columns

    def test_no_nulls_after_cleaning(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert result.isnull().sum().sum() == 0

    def test_drops_null_rows(self):
        df = pd.DataFrame({
            "BHK": [2, np.nan],
            "Rent": [10000, 20000],
            "Size": [800, 900],
            "Floor": ["1 out of 3", "2 out of 4"],
            "Area Type": ["Super Area", "Carpet Area"],
            "Area Locality": ["Whitefield", "Koramangala"],
            "City": ["Bangalore", "Chennai"],
            "Furnishing Status": ["Furnished", "Unfurnished"],
            "Tenant Preferred": ["Family", "Bachelors"],
            "Bathroom": [2, 3],
        })
        result = clean_data(df)
        assert len(result) == 1

    def test_output_shape(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        # Should have: BHK, Rent, Size, Area Type, Area Locality, City,
        # Furnishing Status, Tenant Preferred, Bathroom, floor_num, total_floors
        assert result.shape[1] == 11


# ── validate_data tests ──────────────────────────────────────────────

class TestValidateData:
    def test_valid_data_passes(self, sample_cleaned_df, config):
        # Should not raise
        validate_data(sample_cleaned_df, config)

    def test_missing_target_fails(self, sample_cleaned_df, config):
        df = sample_cleaned_df.drop(columns=["Rent"])
        with pytest.raises(ValueError, match="Target column"):
            validate_data(df, config)


# ── target_encode tests ──────────────────────────────────────────────

class TestTargetEncode:
    def test_output_is_numeric(self, sample_cleaned_df):
        encoded, _ = target_encode(sample_cleaned_df, "Area Locality", "Rent")
        assert pd.api.types.is_numeric_dtype(encoded)

    def test_returns_encoding_map(self, sample_cleaned_df):
        _, encoding_map = target_encode(sample_cleaned_df, "Area Locality", "Rent")
        assert isinstance(encoding_map, dict)
        assert "Whitefield" in encoding_map

    def test_encoded_length_matches(self, sample_cleaned_df):
        encoded, _ = target_encode(sample_cleaned_df, "Area Locality", "Rent")
        assert len(encoded) == len(sample_cleaned_df)


# ── build_features tests ─────────────────────────────────────────────

class TestBuildFeatures:
    def test_output_shapes(self, sample_cleaned_df, config):
        X_train, X_test, y_train, y_test, preprocessor, maps = build_features(sample_cleaned_df, config)
        # Test that train/test split works correctly (using integer division for exact match)
        total_len = len(sample_cleaned_df)
        expected_train = int(total_len * 0.8)
        expected_test = total_len - expected_train
        
        assert X_train.shape[0] == expected_train
        assert X_test.shape[0] == expected_test
        assert len(y_train) == expected_train
        assert len(y_test) == expected_test

    def test_preprocessor_is_fitted(self, sample_cleaned_df, config):
        _, _, _, _, preprocessor, _ = build_features(sample_cleaned_df, config)
        # ColumnTransformer should have transformers_ after fitting
        assert hasattr(preprocessor, "transformers_")

    def test_y_is_rent(self, sample_cleaned_df, config):
        _, _, y_train, y_test, _, _ = build_features(sample_cleaned_df, config)
        # Check that y values are from the Rent column
        assert all(y in sample_cleaned_df["Rent"].values for y in y_train)
        assert all(y in sample_cleaned_df["Rent"].values for y in y_test)

    def test_target_encoding_maps_returned(self, sample_cleaned_df, config):
        _, _, _, _, _, maps = build_features(sample_cleaned_df, config)
        assert "Area Locality" in maps
