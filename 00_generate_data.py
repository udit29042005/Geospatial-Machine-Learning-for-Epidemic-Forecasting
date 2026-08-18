"""
Geospatial ML for Epidemic Forecasting
generate_data.py — builds a synthetic outbreak dataset

Columns:
    region_id     : unique region identifier
    latitude      : region latitude
    longitude     : region longitude
    population    : region population
    confirmed_cases : total confirmed cases
    fatalities    : total deaths (some values are missing on purpose,
                     to be imputed downstream using KNN)
    CFR           : case fatality rate = fatalities / confirmed_cases
                     (computed only where fatalities is known)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_REGIONS = 300

# ---- Simulate several geographic "outbreak clusters" ----
# Each cluster has its own center (lat, lon) and its own severity profile,
# so KMeans on the final data has real structure to recover.
cluster_centers = [
    {"lat": 19.0760, "lon": 72.8777, "severity": 0.045},  # Mumbai-ish, high severity
    {"lat": 28.7041, "lon": 77.1025, "severity": 0.030},  # Delhi-ish, medium
    {"lat": 13.0827, "lon": 80.2707, "severity": 0.020},  # Chennai-ish, lower
    {"lat": 22.5726, "lon": 88.3639, "severity": 0.038},  # Kolkata-ish, medium-high
    {"lat": 12.9716, "lon": 77.5946, "severity": 0.015},  # Bangalore-ish, low
    {"lat": 23.0225, "lon": 72.5714, "severity": 0.050},  # Ahmedabad-ish, high
]

rows = []
regions_per_cluster = N_REGIONS // len(cluster_centers)

region_id = 1
for cluster in cluster_centers:
    for _ in range(regions_per_cluster):
        lat = cluster["lat"] + np.random.normal(0, 0.8)
        lon = cluster["lon"] + np.random.normal(0, 0.8)

        population = int(np.random.uniform(50_000, 2_000_000))
        confirmed_cases = int(population * np.random.uniform(0.01, 0.08))

        true_cfr = max(0.001, np.random.normal(cluster["severity"], 0.01))
        fatalities = int(confirmed_cases * true_cfr)

        rows.append({
            "region_id": f"R{region_id:04d}",
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "population": population,
            "confirmed_cases": confirmed_cases,
            "fatalities": fatalities,
        })
        region_id += 1

df = pd.DataFrame(rows)

# ---- Introduce missing fatalities (~15% of rows) to simulate real-world gaps ----
missing_frac = 0.15
missing_idx = df.sample(frac=missing_frac, random_state=42).index
df.loc[missing_idx, "fatalities"] = np.nan

# ---- CFR is only computable where fatalities is known ----
df["CFR"] = df["fatalities"] / df["confirmed_cases"]

df.to_csv("outbreak_data.csv", index=False)
print(f"Saved outbreak_data.csv with {len(df)} regions "
      f"({df['fatalities'].isna().sum()} missing fatality values).")
print(df.head())
