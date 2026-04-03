# MLOps Price Prediction Pipeline

[![ML Pipeline](https://github.com/Tanx-123/MLOps_Price_Prediction/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/Tanx-123/MLOps_Price_Prediction/actions)
[![Python](https://img.shields.io/badge/Python-3.13+-blue?style=flat&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

End-to-end ML pipeline for predicting rental prices. Think Zillow but for the Indian rental market - pull data, train models, validate quality, deploy to API. Runs on GitHub Actions with AWS for storage.

Started as a learning project but turned into something actually useful. The core idea: automate everything so you can focus on improving the model rather than babysitting pipelines.

## 🚀 What's in the box

- **Data pipeline** - Fetch from S3, clean, encode, split. Standard stuff but it works.
- **4 models compared** - ExtraTrees, RandomForest, LightGBM, XGBoost. ExtraTrees usually wins on this dataset.
- **Hyperparam tuning** - Optional `--optimize` flag uses RandomizedSearchCV. Takes longer but finds better params.
- **Ensemble** - Voting and Stacking ensembles available if you want to experiment.
- **S3 storage** - Processed data and trained models live in your S3 bucket.
- **Quality gate** - Won't deploy if R² is below threshold. Fail safe.
- **FastAPI server** - Loads model from S3 on startup, serves predictions.
- **Metrics endpoint** - Prometheus-ready `/metrics` for monitoring.
- **Data validation** - Checks for nulls, outliers, schema drift.
- **Docker** - Build and push to ECR automatically. See CI/CD section.
- **MLflow** - Optional tracking, start the UI separately if you want it.

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
│   ├── validate_data.py      # Data validation
│   └── upload.py             # S3 upload utilities
├── data/
│   ├── raw/                  # Raw local data storage
│   └── processed/            # Processed CSVs
├── artifacts/                # Local cache for trained models/maps
├── configs/                  # YAML pipelines configs
├── tests/                    # Pytest unit tests
└── logs/                     # Prediction logs
```

## 🎯 Quick Start

Run everything locally first to verify it works:

```bash
# 1. Pull and process data
python -m src.data_pipeline

# 2. Check the cleaned data looks right
python -m src.validate_data --stage clean

# 3. Train models
python -m src.train_pipeline

# Want better params? This takes longer but finds them:
python -m src.train_pipeline --optimize

# 4. Validate on held-out test
python -m src.evaluate

# 5. Start the API
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

The first run will download data from S3, so make sure your `.env` is set up.

## 📊 How it performs

ExtraTrees usually wins on this dataset - something about how it handles the categorical encoding we apply to City/Area combinations. With default params you should see:

- **Test R² ~0.65-0.70** - enough to be useful, not great but workable
- **Test RMSE ~25k-30k** - depends on the data distribution that week
- **Test MAE ~11k-12k** - off by about that much on average in rupees

The CI/CD pipeline just ran this and passed threshold, so baseline is established.

## 🌐 API

The prediction server is just a FastAPI wrapper around the trained model. Start it with:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

It'll pull the latest model from S3 on startup.

### Endpoints

- `GET /health` - Returns `{"status": "healthy", ...}` includes model version and S3 connectivity
- `GET /metrics` - Prometheus scrape endpoint
- `POST /predict` - Main prediction endpoint
- `GET /model/info` - Current model metrics and params

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"BHK": 2, "Size": 800, "City": "Bangalore", ...}'
```

### Metrics

The model logs prediction latency and request counts to `/metrics`. Hook this up to Prometheus + Grafana for production monitoring.

## 🔬 MLflow

Optional but useful for experimenting. Start the UI separately:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then pass `--mlflow-uri http://localhost:5000` when training. Will log params, metrics, and the trained model.

## 🧪 Testing

Run the test suite with:

```bash
pytest tests/ -v --tb=short
```

Covers data processing, model training, and the API endpoints. Should pass locally before you push anything.

## 🔄 CI/CD Pipeline

The full ML pipeline runs automatically on GitHub Actions whenever you push to main. Here's how it works.

### What runs when

| Trigger | Runs tests | Trains | Deploys |
|---------|-----------|--------|---------|
| Push to main | Yes | Yes | Yes |
| PR to main | Yes | No | No |
| Manual dispatch | Yes | Yes | Yes |
| Weekly schedule | Yes | Yes | Yes |

We only run the full training → deploy pipeline on main branch or when you manually trigger it. PRs just run tests to catch issues early without wasting compute.

### Setting up AWS credentials

Go to your repo settings → Secrets and variables → Actions, and add these:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY  
AWS_DEFAULT_REGION   # e.g., ap-south-1
```

Make sure your IAM user has S3 and ECR permissions - a user with PowerUserAccess or equivalent should work for local testing.

### Triggering manually

```bash
gh workflow run ml_pipeline.yml
```

Or just hit "Run workflow" from the Actions tab on GitHub.

## 🎯 Quality Gates

The pipeline won't deploy a bad model - we check R² before building Docker.

### The threshold

Set in `configs/config.yaml`:

```yaml
model:
  r2_threshold: 0.55
```

If test R² is below this, the pipeline stops and nothing gets deployed. Default is 0.55 which is lenient - you're getting 0.6-0.7 with ExtraTrees so there's buffer.

### Making it stricter

```yaml
model:
  r2_threshold: 0.65  # will fail sometimes until you tune more
```

Or looser if you're experimenting:

```yaml
model:
  r2_threshold: 0.50  # almost always passes
```

### When things fail

| Job | Why | Fix |
|-----|-----|-----|
| Train | MLflow path error | Rare on CI, check env vars if it happens |
| Evaluate | R² too low | Need more features or better hyperparams |
| Docker build | AWS creds expired | Refresh secrets in repo settings |

## 🚀 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t mlops-price-prediction .

# Run container (localhost only for security)
docker run -p 127.0.0.1:8000:8000 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_DEFAULT_REGION=ap-south-1 \
  mlops-price-prediction
```

### Run Tests in Docker
```bash
docker run --rm mlops-price-prediction pytest tests/ -v
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

If something breaks, open an issue. PRs welcome.
