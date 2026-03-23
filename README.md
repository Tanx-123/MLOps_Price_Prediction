# MLOps Price Prediction Pipeline

A comprehensive machine learning pipeline for predicting rental prices using modern MLOps practices. This project demonstrates best practices for data processing, model training, evaluation, validation, and deployment in a production environment.

## 🚀 Features

- **Complete Data Pipeline**: Automated data fetching from S3, cleaning, and robust feature engineering with strict train/test splitting.
- **Model Comparison**: Compare 5 state-of-the-art tree-based ML models using cross-validation.
- **Hyperparameter Optimization**: Optional randomized search tuning for absolute best performance.
- **Ensemble Methods**: Automated creation of Voting and Stacking ensemble models.
- **Cloud Integration**: AWS S3 storage for tracking processed features and joblib model artifacts.
- **Evaluation Gate**: Automated quality gate evaluating RMSE, MAE, and R² against thresholds prior to deployment.
- **REST API**: FastAPI-based prediction service loading seamlessly from cloud artifacts.
- **Professional Structure**: Clean, modular, production-ready codebase.

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Tanx-123/MLOps_Price_Prediction.git
cd MLOps_Price_Prediction
```

2. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure AWS and environment variables:
Create a `.env` file from the example template and fill in your keys:
```bash
cp .env.example .env
# Edit .env with your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
```

## 🏗️ Project Structure

```
MLOps_Price_Prediction/
├── src/
│   ├── core_utils.py          # Shared utilities (S3, metrics, config)
│   ├── data_pipeline.py       # Data processing & feature engineering
│   ├── train_pipeline.py      # Model training, tuning, comparison
│   ├── evaluate.py           # Model quality gate evaluation
│   ├── serve.py              # FastAPI prediction service
│   └── upload.py             # S3 upload utilities
├── data/
│   ├── raw/                  # Raw local data storage
│   └── processed/            # Processed CSVs
├── artifacts/                # Local cache for trained models/maps
├── configs/                  # YAML pipelines configs
└── tests/                    # Pytest unit tests
```

## 🎯 Quick Start

Run the MLOps pipeline stages sequentially:

### 1. Data Processing Pipeline
```bash
# Fetch, clean, encode, and build features
python -m src.data_pipeline
```

### 2. Model Training Pipeline
```bash
# Compare base models and save the best
python -m src.train_pipeline

# Or run with robust randomized hyperparameter optimization
python -m src.train_pipeline --optimize
```

### 3. Model Evaluation Pipeline
```bash
# Evaluate the saved model against the strictly held-out test split
python -m src.evaluate
```

### 4. Start Prediction API
```bash
# Start the FastAPI server locally
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

## 📊 Model Performance

The pipeline evaluates and compares 5 top estimators across 5-fold cross-validation:

1. **ExtraTreesRegressor** (Current best model)
2. **RandomForestRegressor**
3. **CatBoostRegressor**
4. **LightGBM**
5. **XGBoost**

**Typical Optimal Performance Characteristics (ExtraTrees):**
- Test RMSE: ~46,243
- Test MAE: ~14,678
- Test R²: ~0.48 (Passes configurable pipeline thresholds)

## 🌐 API Endpoints

Once the `uvicorn` server is running on `localhost:8000`, test the endpoints:

### Health Check
```bash
GET /health
```

### Make Predictions
```bash
POST /predict
Content-Type: application/json

{
  "BHK": 2,
  "Size": 800,
  "Area_Type": "Super Area",
  "City": "Bangalore",
  "Furnishing_Status": "Furnished",
  "Tenant_Preferred": "Family",
  "Bathroom": 2,
  "floor_num": 1,
  "total_floors": 3,
  "Area_Locality": "Whitefield"
}
```

### Get Model Info (Metrics + Specs)
```bash
GET /model/info
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t mlops-price-prediction .

# Run container
docker run -p 8000:8000 mlops-price-prediction
```

## 🧪 Testing

The codebase is thoroughly covered by `pytest`:

```bash
# Run all data/model/API tests
python -m pytest tests/ -v --tb=short
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Built with ❤️ for the ML community**