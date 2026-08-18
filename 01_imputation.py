"""
Geospatial ML for Epidemic Forecasting
01_imputation.py — KNN imputation to estimate missing fatalities
using geolocation and CFR
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("outbreak_data.csv")

print("Missing values before imputation:")
print(df.isnull().sum())

# -----------------------------
# CFR is undefined wherever fatalities is missing, so KNNImputer needs to
# fill both columns together, using geolocation as the neighbor-similarity basis.
# -----------------------------
features_for_imputation = ["latitude", "longitude", "confirmed_cases", "fatalities", "CFR"]
impute_df = df[features_for_imputation].copy()

# Scale features so lat/lon and case counts contribute comparably to distance
scaler = StandardScaler()
scaled = scaler.fit_transform(impute_df)

imputer = KNNImputer(n_neighbors=5, weights="distance")
imputed_scaled = imputer.fit_transform(scaled)

imputed = pd.DataFrame(
    scaler.inverse_transform(imputed_scaled),
    columns=features_for_imputation
)

df["fatalities"] = imputed["fatalities"].round().astype(int)
df["CFR"] = df["fatalities"] / df["confirmed_cases"]

print("\nMissing values after imputation:")
print(df.isnull().sum())

df.to_csv("outbreak_data_imputed.csv", index=False)
print("\nSaved outbreak_data_imputed.csv")
