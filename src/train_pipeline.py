"""
Training pipeline — compare models, optionally tune & ensemble, save best.

Usage:
    python -m src.train_pipeline
    python -m src.train_pipeline --optimize --ensemble
"""
import os
import sys
import json
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import randint, uniform
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.core_utils import (
    load_config, download_from_s3, upload_directory_to_s3,
    build_features, compute_metrics, save_json, save_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Model definitions ────────────────────────────────────────────────

MODELS = {
    "RandomForest": lambda: RandomForestRegressor(random_state=42, n_jobs=-1),
    "XGBoost": lambda: xgb.XGBRegressor(random_state=42, n_jobs=-1),
    "LightGBM": lambda: lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    "CatBoost": lambda: CatBoostRegressor(random_state=42, verbose=False),
    "ExtraTrees": lambda: ExtraTreesRegressor(random_state=42, n_jobs=-1),
}

PARAM_DISTRIBUTIONS = {
    "RandomForest": {
        "n_estimators": randint(100, 300),
        "max_depth": [10, 15, 20, None],
        "min_samples_split": randint(2, 8),
        "min_samples_leaf": randint(1, 4),
    },
    "XGBoost": {
        "n_estimators": randint(100, 300),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.15),
        "subsample": uniform(0.7, 0.3),
    },
    "LightGBM": {
        "n_estimators": randint(100, 300),
        "max_depth": [3, 5, 7, -1],
        "learning_rate": uniform(0.01, 0.15),
        "num_leaves": randint(20, 80),
    },
    "CatBoost": {
        "iterations": randint(100, 300),
        "depth": randint(4, 8),
        "learning_rate": uniform(0.01, 0.15),
    },
    "ExtraTrees": {
        "n_estimators": randint(100, 300),
        "max_depth": [10, 15, 20, None],
        "min_samples_split": randint(2, 8),
        "min_samples_leaf": randint(1, 4),
    },
}


def get_models():
    """Fresh model instances (needed because sklearn mutates models on fit)."""
    return {name: factory() for name, factory in MODELS.items()}


# ── Model comparison ─────────────────────────────────────────────────

def compare_models(X_train, y_train, cv_folds=5):
    """Cross-validate all models and return results sorted by RMSE."""
    models = get_models()
    logger.info(f"Comparing {len(models)} models with {cv_folds}-fold CV...")

    results = {}
    for name, model in models.items():
        try:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds, scoring="neg_mean_squared_error", n_jobs=-1,
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

def optimize_models(X_train, y_train, top_model_names, cv_folds=5, n_iter=20):
    """Run RandomizedSearchCV on the top models. Returns dict with models/scores/params."""
    logger.info("Starting hyperparameter optimization...")
    fresh_models = get_models()

    results = {"models": {}, "scores": {}, "params": {}}
    for name in top_model_names:
        if name not in PARAM_DISTRIBUTIONS:
            logger.warning(f"No param distribution for {name}, skipping")
            continue

        logger.info(f"  Tuning {name} ({n_iter} iterations)...")
        try:
            search = RandomizedSearchCV(
                estimator=fresh_models[name],
                param_distributions=PARAM_DISTRIBUTIONS[name],
                n_iter=n_iter, cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1, random_state=42, verbose=0,
            )
            search.fit(X_train, y_train)

            results["models"][name] = search.best_estimator_
            results["scores"][name] = search.best_score_
            results["params"][name] = search.best_params_
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


def evaluate_ensemble(ensemble, X_train, y_train, X_test, y_test, name):
    """Fit ensemble, compute CV + test metrics."""
    cv_scores = cross_val_score(
        ensemble, X_train, y_train,
        cv=5, scoring="neg_mean_squared_error", n_jobs=-1,
    )
    cv_rmse = np.sqrt(-cv_scores)

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
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

def train_best_model(X_train, X_test, y_train, y_test, best_name, model):
    """Train the given model on full train set and compute metrics."""
    logger.info(f"Training best model: {best_name}")
    model.fit(X_train, y_train)

    train_metrics = compute_metrics(y_train, model.predict(X_train))
    test_metrics = compute_metrics(y_test, model.predict(X_test))

    metrics = {
        "best_model_type": best_name,
        "train": train_metrics,
        "test": test_metrics,
        "cv_results": {},
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
    args = parser.parse_args()

    config = load_config(args.config)
    s3_cfg = config["s3"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    # 1. Load cleaned data
    processed_dir = data_cfg["processed_path"]
    os.makedirs(processed_dir, exist_ok=True)
    clean_path = os.path.join(processed_dir, "cleaned_data.csv")

    if not os.path.exists(clean_path):
        logger.info("Cleaned data not found locally, downloading from S3...")
        if not download_from_s3(s3_cfg["bucket"], s3_cfg["processed_key"], clean_path):
            logger.error("No cleaned data available.")
            sys.exit(1)

    # 2. Preprocess
    logger.info("Preprocessing data...")
    df = pd.read_csv(clean_path)
    X_train, X_test, y_train, y_test, preprocessor, encoding_maps = build_features(df, config)

    # 3. Compare models
    logger.info("Comparing models...")
    cv_results = compare_models(X_train, y_train, cv_folds=args.cv_folds)
    if not cv_results:
        logger.error("No models trained successfully.")
        sys.exit(1)

    top_models = get_top_models(cv_results, top_n=model_cfg["ensemble"]["top_models"])

    # 4. Optional: hyperparameter optimization
    optimized_models = {}
    if args.optimize:
        logger.info("Optimizing hyperparameters...")
        opt_results = optimize_models(X_train, y_train, top_models, args.cv_folds, args.n_iter)
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
            evaluate_ensemble(ensemble, X_train, y_train, X_test, y_test, name)

    # 6. Train best individual model
    logger.info("Training best model...")
    if args.optimize and optimized_models:
        # opt_results["scores"] contains neg_mean_squared_error
        best_name = max(opt_results["scores"], key=lambda k: opt_results["scores"][k])
        model_to_train = optimized_models[best_name]
    else:
        best_name = min(top_models, key=lambda k: top_models[k]["mean_score"])
        model_to_train = get_models()[best_name]

    best_model, metrics = train_best_model(X_train, X_test, y_train, y_test, best_name, model_to_train)

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

    logger.info(f"\nTo start serving: uvicorn src.serve:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()