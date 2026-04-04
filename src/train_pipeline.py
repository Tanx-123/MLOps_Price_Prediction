"""
Training pipeline — compare models, optionally tune & ensemble, save best.

Features:
- MLflow experiment tracking
- Model comparison with cross-validation
- Hyperparameter optimization
- Ensemble methods

Usage:
    python -m src.train_pipeline
    python -m src.train_pipeline --optimize --ensemble
"""
import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import xgboost as xgb
import lightgbm as lgb

from src.core_utils import (
    load_config, download_from_s3, upload_directory_to_s3,
    build_features, compute_metrics, save_json, save_model, TargetEncodingTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── MLflow Setup ───────────────────────────────────────────────────────

def setup_mlflow(experiment_name: str = "rent-prediction", tracking_uri: str = None):
    """Initialize MLflow tracking."""
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping experiment tracking")
        return None
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")
    return mlflow


def log_model_run(model_name: str, model, metrics: dict, params: dict = None):
    """Log model metrics and parameters to MLflow."""
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping model logging")
        return
    
    try:
        artifact_path = os.environ.get("MLFLOW_ARTIFACT_ROOT", "./mlruns")
        
        with mlflow.start_run(run_name=model_name):
            if params:
                mlflow.log_params(params)
            
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        mlflow.log_metric(f"{key}_{subkey}", subvalue)
                else:
                    mlflow.log_metric(key, value)
            
            mlflow.sklearn.log_model(model, model_name, artifact_path=artifact_path)
            logger.info(f"Logged {model_name} to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}. Continuing without model logging.")


# ── Model definitions ────────────────────────────────────────────────

def get_models(config=None):
    """Fresh model instances with config overrides applied."""
    defaults = {}
    if config and "model_defaults" in config:
        defaults = config["model_defaults"]
    
    rf_params = {"random_state": 42, "n_jobs": -1}
    rf_params.update(defaults.get("RandomForest", {}))
    
    xgb_params = {"reg_alpha": 1.0, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1}
    xgb_params.update(defaults.get("XGBoost", {}))
    
    lgb_params = {"reg_alpha": 1.0, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1, "verbose": -1}
    lgb_params.update(defaults.get("LightGBM", {}))
    
    et_params = {"random_state": 42, "n_jobs": -1}
    et_params.update(defaults.get("ExtraTrees", {}))
    
    return {
        "RandomForest": RandomForestRegressor(**rf_params),
        "XGBoost": xgb.XGBRegressor(**xgb_params),
        "LightGBM": lgb.LGBMRegressor(**lgb_params),
        "ExtraTrees": ExtraTreesRegressor(**et_params),
    }

PARAM_DISTRIBUTIONS = {
    "RandomForest": {
        "n_estimators": randint(150, 500),
        "max_depth": [8, 10, 15, 20, None],
        "min_samples_split": randint(5, 20),
        "min_samples_leaf": randint(3, 10),
        "max_features": ["sqrt", "log2", 0.5, 0.7],
    },
    "XGBoost": {
        "n_estimators": randint(150, 500),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_alpha": uniform(0.0, 5.0),
        "reg_lambda": uniform(0.5, 5.0),
    },
    "LightGBM": {
        "n_estimators": randint(150, 500),
        "max_depth": [3, 5, 7, 10, -1],
        "learning_rate": uniform(0.01, 0.19),
        "num_leaves": randint(15, 100),
        "subsample": uniform(0.6, 0.4),
        "reg_alpha": uniform(0.0, 5.0),
        "reg_lambda": uniform(0.5, 5.0),
    },
    "ExtraTrees": {
        "n_estimators": randint(150, 500),
        "max_depth": [8, 10, 15, 20, None],
        "min_samples_split": randint(5, 20),
        "min_samples_leaf": randint(3, 10),
        "max_features": ["sqrt", "log2", 0.5, 0.7],
    },
}


def _make_feature_preprocessor(config, df=None):
    features = config["features"]
    high_cardinality = features.get("high_cardinality", [])
    
    if df is not None:
        num_cols = [c for c in features["numerical"] + high_cardinality if c in df.columns]
        cat_cols = [c for c in features["categorical"] if c in df.columns]
    else:
        num_cols = features["numerical"] + high_cardinality
        cat_cols = features["categorical"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def _make_cv_pipeline(model, config, smoothing=None, train_df=None):
    features = config["features"]
    high_cardinality = features.get("high_cardinality", [])
    if smoothing is None:
        smoothing = features.get("target_encoding_smoothing", 10)

    return Pipeline(
        steps=[
            ("target_encode", TargetEncodingTransformer(columns=high_cardinality, smoothing=smoothing)),
            ("preprocessor", _make_feature_preprocessor(config, train_df)),
            ("model", model),
        ]
    )


# ── Model comparison ─────────────────────────────────────────────────

def compare_models(train_df, config, cv_folds=5):
    """Cross-validate all models and return results sorted by RMSE."""
    models = get_models(config)
    logger.info(f"Comparing {len(models)} models with {cv_folds}-fold CV...")

    results = {}
    target_col = config["features"]["target"]
    target_transform = config["features"].get("target_transform", None)
    y_train = train_df[target_col].values
    if target_transform == "log1p":
        y_train = np.log1p(y_train)
    for name, model in models.items():
        try:
            pipe = _make_cv_pipeline(model, config, train_df=train_df)
            scores = cross_val_score(
                pipe,
                train_df,
                y_train,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            rmse_scores = np.sqrt(-scores)
            results[name] = {
                "mean_score": np.mean(rmse_scores),
                "std_score": np.std(rmse_scores),
                "cv_scores": rmse_scores.tolist(),
                "scoring": "RMSE",
            }
            logger.info(f"  {name}: RMSE = {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores) * 2:.4f})")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")

    return results


def get_top_models(results, top_n=3):
    """Return the top-N model names by lowest mean RMSE."""
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_score"])
    top = dict(ranked[:top_n])
    logger.info(f"Top {top_n}: {list(top.keys())}")
    return top


# ── Hyperparameter tuning ────────────────────────────────────────────

def optimize_models(train_df, config, top_model_names, cv_folds=5, n_iter=20):
    """Run RandomizedSearchCV on the top models. Returns dict with models/scores/params."""
    logger.info("Starting hyperparameter optimization...")
    fresh_models = get_models(config)

    results = {"models": {}, "scores": {}, "params": {}}
    target_col = config["features"]["target"]
    y_train = train_df[target_col].values
    for name in top_model_names:
        if name not in PARAM_DISTRIBUTIONS:
            logger.warning(f"No param distribution for {name}, skipping")
            continue

        logger.info(f"  Tuning {name} ({n_iter} iterations)...")
        try:
            base_model = fresh_models[name]
            pipe = _make_cv_pipeline(base_model, config, train_df=train_df)

            # Our pipeline has the regressor at `model`, so param keys must be `model__<param>`.
            param_dist = {f"model__{k}": v for k, v in PARAM_DISTRIBUTIONS[name].items()}

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1, random_state=42, verbose=0,
            )
            search.fit(train_df, y_train)

            results["scores"][name] = float(search.best_score_)
            results["params"][name] = search.best_params_

            # Store just the tuned regressor (we'll train it later on preprocessed X_train).
            tuned_model = get_models(config)[name]
            stripped_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            tuned_model.set_params(**stripped_params)
            results["models"][name] = tuned_model

            logger.info(f"  {name}: best neg MSE = {search.best_score_:.4f}")
        except Exception as e:
            logger.error(f"  {name} optimization failed: {e}")

    return results


# ── Ensembles ────────────────────────────────────────────────────────

def _make_estimator_list(models, max_n=3):
    """Convert {name: model} → [(name, model), ...] capped at max_n."""
    return [(n.lower(), m) for n, m in list(models.items())[:max_n]]


def create_voting_ensemble(optimized_models):
    estimators = _make_estimator_list(optimized_models)
    logger.info(f"Voting ensemble with: {[n for n, _ in estimators]}")
    return VotingRegressor(estimators=estimators, n_jobs=-1)


def create_stacking_ensemble(optimized_models, cv_folds=5):
    estimators = _make_estimator_list(optimized_models)
    meta = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    logger.info(f"Stacking ensemble with: {[n for n, _ in estimators]}, meta=RandomForest")
    return StackingRegressor(
        estimators=estimators, final_estimator=meta,
        cv=cv_folds, n_jobs=-1, verbose=0,
    )


def evaluate_ensemble(ensemble, train_df, test_df, config, name, cv_folds=5):
    """Fit ensemble pipeline, compute CV + test metrics."""
    target_col = config["features"]["target"]
    target_transform = config["features"].get("target_transform", None)
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    if target_transform == "log1p":
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    pipe = _make_cv_pipeline(ensemble, config, train_df=train_df)

    cv_scores = cross_val_score(
        pipe,
        train_df,
        y_train,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    cv_rmse = np.sqrt(-cv_scores)

    pipe.fit(train_df, y_train)
    y_pred = pipe.predict(test_df)
    metrics = compute_metrics(y_test, y_pred)

    result = {
        "ensemble_name": name,
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_rmse_std": float(np.std(cv_rmse)),
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
        "test_r2": metrics["r2"],
        "cv_scores": cv_rmse.tolist(),
    }
    logger.info(f"  {name}: CV RMSE={np.mean(cv_rmse):.4f}, Test RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    return result


# ── Train best individual model ──────────────────────────────────────

def train_best_model(X_train, X_test, y_train, y_test, best_name, model, config):
    """Train the given model on full train set and compute metrics.

    If target was log-transformed, inverse-transforms predictions for real-scale metrics.
    """
    logger.info(f"Training best model: {best_name}")
    model.fit(X_train, y_train)

    target_transform = config["features"].get("target_transform", None)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    y_train_real = y_train
    y_test_real = y_test

    # Inverse-transform for real-scale metrics
    if target_transform == "log1p":
        train_pred = np.expm1(train_pred)
        test_pred = np.expm1(test_pred)
        y_train_real = np.expm1(y_train)
        y_test_real = np.expm1(y_test)

    train_metrics = compute_metrics(y_train_real, train_pred)
    test_metrics = compute_metrics(y_test_real, test_pred)

    metrics = {
        "best_model_type": best_name,
        "train": train_metrics,
        "test": test_metrics,
        "cv_results": {},
        "target_transform": target_transform or "none",
    }
    logger.info(f"  Train: {train_metrics}")
    logger.info(f"  Test:  {test_metrics}")

    return model, metrics


# ── Main pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--ensemble", action="store_true", help="Build ensemble models")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    args = parser.parse_args()

    config = load_config(args.config)
    s3_cfg = config["s3"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    # Setup MLflow
    setup_mlflow(tracking_uri=args.mlflow_uri)

    # 1. Load cleaned data
    processed_dir = data_cfg["processed_path"]
    os.makedirs(processed_dir, exist_ok=True)
    clean_path = os.path.join(processed_dir, "cleaned_data.csv")

    if not os.path.exists(clean_path):
        logger.info("Cleaned data not found locally, downloading from S3...")
        if not download_from_s3(s3_cfg["bucket"], s3_cfg["processed_key"], clean_path):
            logger.error("No cleaned data available.")
            sys.exit(1)

    # 2. Load cleaned data and compute the outer split used for CV + final training.
    logger.info("Loading cleaned data...")
    df = pd.read_csv(clean_path)
    test_size = model_cfg.get("test_size", 0.2)
    random_state = model_cfg.get("random_state", 42)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Preprocess for final (outer) train/test fit + artifact saving.
    logger.info("Preprocessing data for final training...")
    X_train, X_test, y_train, y_test, preprocessor, encoding_maps = build_features(df, config)

    # 3. Compare models
    logger.info("Comparing models...")
    cv_results = compare_models(train_df, config, cv_folds=args.cv_folds)
    if not cv_results:
        logger.error("No models trained successfully.")
        sys.exit(1)

    top_models = get_top_models(cv_results, top_n=model_cfg["ensemble"]["top_models"])

    # 4. Optional: hyperparameter optimization
    optimized_models = {}
    if args.optimize:
        logger.info("Optimizing hyperparameters...")
        opt_results = optimize_models(train_df, config, top_models, args.cv_folds, args.n_iter)
        optimized_models = opt_results["models"]

        # Save optimization results
        opt_path = os.path.join(data_cfg["artifacts_path"], "hyperparameter_optimization.json")
        save_json({
            "scores": {k: float(v) for k, v in opt_results["scores"].items()},
            "params": opt_results["params"],
        }, opt_path)

    # 5. Optional: ensemble creation
    if args.ensemble and optimized_models:
        logger.info("Building ensembles...")
        for factory, name in [
            (create_voting_ensemble, "Voting Ensemble"),
            (lambda m: create_stacking_ensemble(m, args.cv_folds), "Stacking Ensemble"),
        ]:
            ensemble = factory(optimized_models)
            evaluate_ensemble(ensemble, train_df, test_df, config, name)

    # 6. Train best individual model
    logger.info("Training best model...")
    if args.optimize and optimized_models:
        # opt_results["scores"] contains neg_mean_squared_error
        best_name = max(opt_results["scores"], key=lambda k: opt_results["scores"][k])
        model_to_train = optimized_models[best_name]
    else:
        best_name = min(top_models, key=lambda k: top_models[k]["mean_score"])
        model_to_train = get_models(config)[best_name]

    best_model, metrics = train_best_model(X_train, X_test, y_train, y_test, best_name, model_to_train, config)

    # Log to MLflow
    log_model_run(
        model_name=best_name,
        model=best_model,
        metrics={
            "train_rmse": metrics["train"]["rmse"],
            "train_mae": metrics["train"]["mae"],
            "train_r2": metrics["train"]["r2"],
            "test_rmse": metrics["test"]["rmse"],
            "test_mae": metrics["test"]["mae"],
            "test_r2": metrics["test"]["r2"],
        },
        params={"model_type": best_name, "target_transform": metrics.get("target_transform", "none")}
    )

    # 7. Save artifacts
    artifacts_dir = data_cfg["artifacts_path"]
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "best_model.joblib")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    encoding_path = os.path.join(artifacts_dir, "target_encoding_maps.joblib")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")

    save_model(best_model, model_path)
    save_model(preprocessor, preprocessor_path)
    save_model(encoding_maps, encoding_path)
    save_json(metrics, metrics_path)
    logger.info(f"Saved artifacts to {artifacts_dir}/")

    # 8. Upload to S3
    logger.info("Uploading artifacts to S3...")
    if not upload_directory_to_s3(artifacts_dir, s3_cfg["bucket"], s3_cfg["artifacts_prefix"]):
        logger.error("S3 upload failed.")
        sys.exit(1)

    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Model: {best_model.__class__.__name__}")
    logger.info(f"Test RMSE: {metrics['test']['rmse']:.4f}  MAE: {metrics['test']['mae']:.4f}  R²: {metrics['test']['r2']:.4f}")

    for name in sorted(cv_results, key=lambda k: cv_results[k]["mean_score"]):
        r = cv_results[name]
        logger.info(f"  {name:15}: CV RMSE = {r['mean_score']:.4f} (±{r['std_score'] * 2:.4f})")

    logger.info("\nTo start serving: uvicorn src.serve:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()