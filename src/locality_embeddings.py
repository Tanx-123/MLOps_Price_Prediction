"""
Locality embedding generator — converts locality names to dense vectors.
Uses sentence-transformers to encode locality names, then reduces dimensions with PCA.

Usage:
    python -m src.locality_embeddings
"""
import os
import sys
import logging
import argparse

import pandas as pd
import numpy as np
import joblib

from src.core_utils import load_config, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CITY_COORDINATES = {
    "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "hyderabad": (17.3850, 78.4867),
    "delhi": (28.7041, 77.1025),
    "kolkata": (22.5726, 88.3639),
}


def generate_embeddings(localities, model_name, dimensions):
    """Generate embeddings for a list of locality strings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Generating embeddings for {len(localities)} localities...")
    embeddings = model.encode(localities, show_progress_bar=True)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    return embeddings, model


def reduce_dimensions(embeddings, n_components, random_state=42):
    """Reduce embedding dimensions using PCA."""
    from sklearn.decomposition import PCA

    if embeddings.shape[1] <= n_components:
        logger.warning(f"Original dimensions ({embeddings.shape[1]}) <= target ({n_components}), skipping PCA")
        return embeddings, None

    logger.info(f"Reducing dimensions from {embeddings.shape[1]} to {n_components}")
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")
    return reduced, pca


def add_city_coordinates(df):
    """Add city_lat and city_lon based on City column."""
    df = df.copy()
    df["city_lat"] = df["City"].str.lower().map(lambda x: CITY_COORDINATES.get(x, (0, 0))[0])
    df["city_lon"] = df["City"].str.lower().map(lambda x: CITY_COORDINATES.get(x, (0, 0))[1])
    logger.info("Added city_lat and city_lon features")
    return df


def generate_locality_embeddings(df, config):
    """Main function to generate and save locality embeddings."""
    location_config = config.get("location", {})
    embedding_config = location_config.get("embedding", {})
    cache_path = embedding_config.get("cache_path", "artifacts/locality_embeddings.joblib")

    if os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        cached = joblib.load(cache_path)
        return cached["embeddings_map"], cached["pca"]

    model_name = embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    dimensions = embedding_config.get("dimensions", 16)

    unique_city_localities = df["City_Locality"].unique().tolist()
    embeddings, model = generate_embeddings(unique_city_localities, model_name, dimensions)

    reduced_embeddings, pca = reduce_dimensions(embeddings, dimensions)

    embeddings_map = {loc: reduced_embeddings[i] for i, loc in enumerate(unique_city_localities)}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    artifacts = {
        "embeddings_map": embeddings_map,
        "pca": pca,
        "model_name": model_name,
        "dimensions": dimensions,
    }
    save_model(artifacts, cache_path)
    logger.info(f"Saved embeddings to {cache_path}")

    return embeddings_map, pca


def apply_locality_embeddings(df, embeddings_map, n_components):
    """Apply pre-computed embeddings to dataframe using City_Locality key."""
    dim = n_components
    embedding_cols = [f"locality_emb_{i}" for i in range(dim)]

    embedding_matrix = np.zeros((len(df), dim))
    for i, key in enumerate(df["City_Locality"]):
        if key in embeddings_map:
            embedding_matrix[i] = embeddings_map[key]

    for j, col in enumerate(embedding_cols):
        df[col] = embedding_matrix[:, j]

    logger.info(f"Added {dim} locality embedding features")
    return df


def generate_localities_json(df, output_path="data/processed/localities_by_city.json"):
    """Generate JSON mapping of cities to their localities for frontend."""
    localities_by_city = {}
    for city in df["City"].unique():
        city_data = df[df["City"] == city]
        localities = sorted(city_data["Area Locality"].unique().tolist())
        localities_by_city[city.capitalize()] = localities

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import json
    with open(output_path, "w") as f:
        json.dump(localities_by_city, f, indent=2)
    logger.info(f"Saved localities JSON to {output_path}")
    return localities_by_city


def main():
    parser = argparse.ArgumentParser(description="Generate locality embeddings")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="data/processed/cleaned_data.csv")
    args = parser.parse_args()

    config = load_config(args.config)

    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        logger.info("Run data pipeline first: python -m src.data_pipeline --skip-download")
        sys.exit(1)

    df = pd.read_csv(args.data)
    logger.info(f"Loaded data: {df.shape}")

    df = add_city_coordinates(df)
    embeddings_map, pca = generate_locality_embeddings(df, config)

    dim = config.get("location", {}).get("embedding", {}).get("dimensions", 16)
    df = apply_locality_embeddings(df, embeddings_map, dim)

    generate_localities_json(df)

    output_path = args.data
    df.to_csv(output_path, index=False)
    logger.info(f"Saved enhanced data to {output_path}")


if __name__ == "__main__":
    main()
