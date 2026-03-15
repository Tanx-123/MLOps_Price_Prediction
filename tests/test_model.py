"""
Tests for model training and FastAPI serving endpoints.
"""

import os
import json
import pytest
import numpy as np
import joblib
from unittest.mock import patch
from fastapi.testclient import TestClient
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def artifacts_dir(tmp_path):
    """Create a temporary artifacts directory with model and preprocessor."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    import pandas as pd

    # Create and save a preprocessor
    numerical_cols = ["BHK", "Size", "Bathroom", "floor_num", "total_floors"]
    categorical_cols = ["Area Type", "City", "Furnishing Status", "Tenant Preferred"]
    high_cardinality_cols = ["Area Locality"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols + high_cardinality_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    # Fit the preprocessor on sample data
    sample_df = pd.DataFrame({
        "BHK": [2, 3, 2, 3],
        "Size": [800, 1200, 900, 1100],
        "Bathroom": [2, 3, 2, 3],
        "floor_num": [1, 0, 2, 1],
        "total_floors": [3, 5, 4, 3],
        "Area Locality": ["Whitefield", "Koramangala", "Whitefield", "Indiranagar"],
        "Area Type": ["Super Area", "Carpet Area", "Super Area", "Carpet Area"],
        "City": ["Bangalore", "Chennai", "Bangalore", "Chennai"],
        "Furnishing Status": ["Furnished", "Semi-Furnished", "Unfurnished", "Furnished"],
        "Tenant Preferred": ["Family", "Bachelors", "Bachelors/Family", "Family"],
        "Rent": [10000, 25000, 15000, 20000],
    })
    
    # Apply target encoding for high cardinality features
    from src.core_utils import target_encode
    sample_df["Area Locality"], encoding_map = target_encode(sample_df, "Area Locality", "Rent")
    
    X_transformed = preprocessor.fit_transform(sample_df)
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")

    # Train model on preprocessor output so feature counts match
    y = np.array([10000, 25000, 15000, 20000])
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_transformed, y)
    joblib.dump(model, tmp_path / "best_model.joblib")

    # Save target encoding maps
    joblib.dump({"Area Locality": encoding_map}, tmp_path / "target_encoding_maps.joblib")

    # Save metrics
    metrics = {
        "train": {"mae": 100.0, "rmse": 150.0, "r2": 0.95},
        "test": {"mae": 200.0, "rmse": 300.0, "r2": 0.85},
        "model_type": "RandomForestRegressor",
    }
    with open(tmp_path / "metrics.json", "w") as f:
        json.dump(metrics, f)

    return str(tmp_path)


# ── Model tests ──────────────────────────────────────────────────────

class TestModel:
    def test_model_loads(self, artifacts_dir):
        model = joblib.load(os.path.join(artifacts_dir, "best_model.joblib"))
        assert model is not None

    def test_model_predicts(self, artifacts_dir):
        import pandas as pd
        model = joblib.load(os.path.join(artifacts_dir, "best_model.joblib"))
        preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))
        df = pd.DataFrame({
            "BHK": [2], "Size": [800], "Bathroom": [2],
            "floor_num": [1], "total_floors": [3], "Area Locality": [12500.0],
            "Area Type": ["Super Area"], "City": ["Bangalore"],
            "Furnishing Status": ["Furnished"], "Tenant Preferred": ["Family"],
        })
        X = preprocessor.transform(df)
        pred = model.predict(X)
        assert pred.shape == (1,)

    def test_prediction_is_positive(self, artifacts_dir):
        import pandas as pd
        model = joblib.load(os.path.join(artifacts_dir, "best_model.joblib"))
        preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))
        df = pd.DataFrame({
            "BHK": [2], "Size": [800], "Bathroom": [2],
            "floor_num": [1], "total_floors": [3], "Area Locality": [12500.0],
            "Area Type": ["Super Area"], "City": ["Bangalore"],
            "Furnishing Status": ["Furnished"], "Tenant Preferred": ["Family"],
        })
        X = preprocessor.transform(df)
        pred = model.predict(X)
        assert pred[0] > 0


# ── API endpoint tests ───────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self, artifacts_dir):
        """Create a test client with mocked config."""
        mock_config = {
            "data": {"artifacts_path": artifacts_dir, "processed_path": "data/processed"},
            "s3": {"bucket": "test", "artifacts_prefix": "artifacts", "features_prefix": "features"},
            "model": {"type": "RandomForestRegressor"},
            "server": {"host": "0.0.0.0", "port": 8000},
        }

        with patch("src.serve.load_config", return_value=mock_config):
            from src.serve import app
            # Manually trigger startup
            import asyncio

            with TestClient(app) as client:
                yield client

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_model_info_endpoint(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "RandomForestRegressor"
        assert "metrics" in data

    def test_predict_endpoint(self, client):
        payload = {
            "BHK": 2,
            "Size": 800,
            "Bathroom": 2,
            "floor_num": 1,
            "total_floors": 3,
            "Area_Type": "Super Area",
            "City": "Bangalore",
            "Furnishing_Status": "Furnished",
            "Tenant_Preferred": "Family",
            "Area_Locality": "Whitefield",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_rent" in data
        assert data["predicted_rent"] > 0
        assert data["currency"] == "INR"

    def test_predict_invalid_input(self, client):
        payload = {"BHK": -1}  # Missing required fields
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error