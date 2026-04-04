"""Location utilities - city coordinates and locality data for frontend."""

import os
import logging
import json

logger = logging.getLogger(__name__)

CITY_COORDINATES = {
    "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "hyderabad": (17.3850, 78.4867),
    "delhi": (28.7041, 77.1025),
    "kolkata": (22.5726, 88.3639),
}


def add_city_coordinates(df, config=None):
    """Add city_lat and city_lon based on City column."""
    if config is not None:
        cities = config.get("location", {}).get("cities", {})
        if not cities:
            cities = CITY_COORDINATES
    else:
        cities = CITY_COORDINATES
    
    df = df.copy()
    df["city_lat"] = df["City"].str.lower().map(lambda x: cities.get(x, (0, 0))[0])
    df["city_lon"] = df["City"].str.lower().map(lambda x: cities.get(x, (0, 0))[1])
    logger.info("Added city_lat and city_lon features")
    return df


def generate_localities_json(df, output_path="data/processed/localities_by_city.json"):
    """Generate JSON mapping of cities to their localities for frontend."""
    localities_by_city = {}
    for city in df["City"].unique():
        city_data = df[df["City"] == city]
        raw_localities = city_data["Area Locality"].unique().tolist()
        
        # Filter out invalid entries
        valid_localities = []
        for loc in raw_localities:
            loc_lower = loc.lower().strip()
            # Skip pin codes (5-6 digit numbers)
            if loc_lower.isdigit() and len(loc_lower) in (5, 6):
                continue
            # Skip entries containing "bhk"
            if "bhk" in loc_lower:
                continue
            valid_localities.append(loc)
        
        localities_by_city[city.capitalize()] = sorted(valid_localities)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(localities_by_city, f, indent=2)
    logger.info(f"Saved localities JSON to {output_path}")
    return localities_by_city