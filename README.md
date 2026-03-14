# MLOps Price Prediction Pipeline

A comprehensive machine learning pipeline for predicting rental prices using modern MLOps practices. This project demonstrates best practices for data processing, model training, evaluation, and deployment in a production environment.

## 🚀 Features

- **Complete Data Pipeline**: Automated data fetching, cleaning, and feature engineering
- **Model Comparison**: Compare 10 different ML models using cross-validation
- **Hyperparameter Optimization**: Optional hyperparameter tuning for best performance
- **Ensemble Methods**: Create ensemble models for improved predictions
- **Cloud Integration**: S3 storage for data and model artifacts
- **REST API**: FastAPI-based prediction service
- **Professional Structure**: Clean, maintainable codebase following ML best practices

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

4. Configure AWS credentials (for S3 operations):
```bash
aws configure
```

## 🏗️ Project Structure

```
MLOps_Price_Prediction/
├── src/
│   ├── core_utils.py          # Core utilities and S3 operations
│   ├── data_pipeline.py       # Complete data processing pipeline
│   ├── train_pipeline.py      # Model training and comparison
│   ├── serve.py              # FastAPI prediction service
│   ├── evaluate.py           # Model evaluation utilities
│   └── upload.py             # Simple upload utilities
├── data/
│   ├── raw/                  # Raw data storage
│   └── processed/            # Processed data and features
├── artifacts/                # Trained models and metrics
├── configs/                  # Configuration files
└── tests/                    # Unit tests
```

## 🎯 Quick Start

### 1. Data Processing Pipeline
```bash
# Run complete data pipeline
python -m src.data_pipeline

# Skip download if data already exists
python -m src.data_pipeline --skip-download
```

### 2. Model Training Pipeline
```bash
# Run complete training pipeline
python -m src.train_pipeline

# With hyperparameter optimization
python -m src.train_pipeline --optimize

# With ensemble creation
python -m src.train_pipeline --ensemble

# With both optimization and ensemble
python -m src.train_pipeline --optimize --ensemble
```

### 3. Start Prediction API
```bash
# Start the FastAPI server
uvicorn src.serve:app --host 0.0.0.0 --port 8000

# Or use the serve module
python -m src.serve
```

## 📊 Model Performance

The pipeline compares 10 different models using 5-fold cross-validation:

1. **ExtraTrees**: Best performing model (RMSE ~44,562)
2. **RandomForest**: Strong baseline (RMSE ~45,127)
3. **Ridge**: Linear baseline (RMSE ~45,127)
4. **Lasso**: Regularized linear (RMSE ~45,127)
5. **ElasticNet**: Elastic net regression (RMSE ~45,127)
6. **SVR**: Support Vector Regression (RMSE ~45,127)
7. **MLP**: Neural network (RMSE ~45,127)
8. **CatBoost**: Gradient boosting (RMSE ~45,801)
9. **LightGBM**: Light gradient boosting (RMSE ~46,124)
10. **XGBoost**: Extreme gradient boosting (RMSE ~65,575)

**Best Model Performance:**
- Test RMSE: 42,999
- Test MAE: 14,549
- Test R²: 0.55

## 🔧 Configuration

Configuration is managed through `configs/config.yaml`:

```yaml
data:
  raw_url: "https://raw.githubusercontent.com/amankharwal/Website-data/master/Instagram.csv"
  raw_path: "data/raw/raw_data.csv"
  processed_path: "data/processed/cleaned_data.csv"

s3:
  bucket: "price-trend-tanx"
  region: "ap-south-1"

models:
  hyperparameter_optimization: true
  ensemble_creation: true
  cv_folds: 5
  test_size: 0.2
  random_state: 42

api:
  host: "0.0.0.0"
  port: 8000
```

## 🌐 API Endpoints

Once the server is running, you can make predictions:

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
  "Size": 1200,
  "Area Type": "Super built-up  Area",
  "City": "Mumbai",
  "Furnishing Status": "Semi-Furnished",
  "Tenant Preferred": "Bachelors/Family",
  "Bathroom": 2,
  "floor_num": 5,
  "total_floors": 10,
  "Area Locality": "Andheri"
}
```

### Get Model Info
```bash
GET /model-info
```

## 📈 Data Processing

The data pipeline includes:

1. **Data Download**: Fetch raw data from remote sources
2. **Data Cleaning**: Remove invalid entries and parse complex fields
3. **Feature Engineering**: Create numerical and categorical features
4. **Target Encoding**: Apply target encoding for categorical variables
5. **Train/Test Split**: Proper splitting to prevent data leakage
6. **S3 Upload**: Store processed data and features in cloud storage

## 🤖 Model Training

The training pipeline:

1. **Model Comparison**: Evaluate 10 different algorithms
2. **Cross-Validation**: Use 5-fold CV for robust evaluation
3. **Hyperparameter Tuning**: Optional optimization using GridSearchCV
4. **Ensemble Creation**: Combine top models for better performance
5. **Model Selection**: Choose best performing model
6. **Artifact Storage**: Save models, preprocessors, and metrics
7. **S3 Upload**: Store artifacts in cloud storage

## 🚀 Deployment

### Local Development
```bash
# Start development server
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment
```bash
# Start production server
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```bash
# Build Docker image
docker build -t mlops-price-prediction .

# Run container
docker run -p 8000:8000 mlops-price-prediction
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_features.py

# Run with coverage
python -m pytest tests/ --cov=src
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support, email tanx1234567890@gmail.com or create an issue on GitHub.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [AWS S3](https://aws.amazon.com/s3/) for cloud storage
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [NumPy](https://numpy.org/) for numerical computing

---

**Built with ❤️ for the ML community**