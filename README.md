# MLOps Price Prediction

[![ML Pipeline](https://github.com/Tanx-123/MLOps_Price_Prediction/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/Tanx-123/MLOps_Price_Prediction/actions)

End-to-end ML pipeline for predicting rental prices in the Indian market. Automatically fetches data, trains several models, checks quality, and deploys a FastAPI inference server. We use GitHub Actions for CI/CD and AWS (S3/ECR) for storage and artifacts.

It started as a small script but grew into a full pipeline so I wouldn't have to manually tune and deploy models every time new data comes in.

## Setup

1. **Clone & Virtualenv**
```bash
git clone https://github.com/Tanx-123/MLOps_Price_Prediction.git
cd MLOps_Price_Prediction
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. **Install deps**
```bash
pip install -r requirements.txt
```

3. **Environment setup**
Copy `.env.example` to `.env` and add your AWS credentials:
```bash
cp .env.example .env
# Open .env and add your AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
```

## Running Locally

To test the pipeline locally, run these in order:

```bash
# Fetch and clean the data
python -m src.data_pipeline

# Sanity check the cleaned data
python -m src.validate_data --stage clean

# Train the models and pick the best one
python -m src.train_pipeline

# (Optional) If you want it to run a hyperparameter search, pass --optimize
# python -m src.train_pipeline --optimize

# Evaluate the final model
python -m src.evaluate

# Spin up the API server (loads the current model)
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```
Note: The first run pulls data from S3, so make sure your `.env` is configured correctly.

## API Reference
The prediction server is a standard FastAPI app. It pulls the latest deployed model from S3 on startup.

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

- `GET /health`: Health check, returns model version/status.
- `GET /metrics`: Prometheus metrics.
- `POST /predict`: Main prediction endpoint. Expects JSON with property features.
- `GET /model/info`: Current model parameters and evaluation metrics.

Test the endpoint:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"BHK": 2, "Size": 800, "City": "Bangalore", "Area": "Indiranagar", "Furnishing": "Semi-Furnished", "Tenant_Preferred": "Bachelors/Family", "Bathrooms": 2}'
```

## MLflow Tracking (Optional)

If you're messing around with new models, you can spin up the MLflow UI:
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Then pass `--mlflow-uri http://localhost:5000` when running `train_pipeline.py`.

## CI/CD and Quality Gates

The pipeline runs on GitHub Actions. It is built to automatically reject bad models.

In `configs/config.yaml`, there's an `r2_threshold` (default `0.55`). If the best model's R² on the test set is below this, the pipeline fails and won't deploy. Usually, ExtraTrees gets around 0.65-0.70 here.

Workflow triggers:
- **Push to main / Manual dispatch / Weekly cron:** Runs the full pipeline (train -> eval -> docker build -> deploy).
- **PRs:** Just runs `pytest tests/` to verify nothing is broken.

## Deployment

The CI builds a Docker image and pushes it to ECR. If you need to build and run it manually:

```bash
docker build -t mlops-price-prediction .
docker run -p 8000:8000 --env-file .env mlops-price-prediction
```

## Running Tests

```bash
pytest tests/ -v
```
Make sure tests pass locally before opening a PR.
---
Hit me up with an issue if something is broken. PRs are always welcome.
