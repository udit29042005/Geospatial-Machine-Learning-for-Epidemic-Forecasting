"""
Geospatial ML for Epidemic Forecasting
03_cluster_stats_and_viz.py — engineer spatial features and compute
cluster-wise death and CFR statistics to enhance predictive accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("outbreak_data_clustered.csv")

# -----------------------------
# Additional engineered spatial features
# -----------------------------
# Distance of each region from the overall population-weighted centroid
centroid_lat = np.average(df["latitude"], weights=df["population"])
centroid_lon = np.average(df["longitude"], weights=df["population"])

df["dist_from_centroid"] = np.sqrt(
    (df["latitude"] - centroid_lat) ** 2 + (df["longitude"] - centroid_lon) ** 2
)

# Cases and fatalities per 100k population (normalizes for region size)
df["cases_per_100k"] = df["confirmed_cases"] / df["population"] * 100_000
df["fatalities_per_100k"] = df["fatalities"] / df["population"] * 100_000

# -----------------------------
# Cluster-wise death and CFR statistics
# -----------------------------
cluster_stats = df.groupby("cluster").agg(
    n_regions=("region_id", "count"),
    total_population=("population", "sum"),
    total_cases=("confirmed_cases", "sum"),
    total_fatalities=("fatalities", "sum"),
    mean_CFR=("CFR", "mean"),
    median_CFR=("CFR", "median"),
    mean_fatalities_per_100k=("fatalities_per_100k", "mean"),
    mean_dist_from_centroid=("dist_from_centroid", "mean"),
).round(4)

cluster_stats["overall_CFR"] = (
    cluster_stats["total_fatalities"] / cluster_stats["total_cases"]
).round(4)

print("Cluster-wise death and CFR statistics:\n")
print(cluster_stats)

cluster_stats.to_csv("cluster_statistics.csv")

# -----------------------------
# Visualization: regions colored by cluster, sized by CFR
# -----------------------------
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    df["longitude"], df["latitude"],
    c=df["cluster"], cmap="tab10",
    s=df["CFR"] * 3000,
    alpha=0.7, edgecolors="k", linewidths=0.3
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Regional Clusters (color) sized by CFR")
legend = plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
plt.gca().add_artist(legend)
plt.tight_layout()
plt.savefig("cluster_map.png", dpi=150)
plt.close()

# Bar chart of cluster-wise CFR
plt.figure(figsize=(9, 5))
cluster_stats["overall_CFR"].sort_values().plot(kind="barh", color="crimson")
plt.xlabel("Overall CFR")
plt.ylabel("Cluster")
plt.title("Cluster-wise Case Fatality Rate (CFR)")
plt.tight_layout()
plt.savefig("cluster_cfr_barchart.png", dpi=150)
plt.close()

print("\nSaved cluster_statistics.csv, cluster_map.png, cluster_cfr_barchart.png")
