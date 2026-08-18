"""
Geospatial ML for Epidemic Forecasting
02_clustering.py — KMeans clustering on geospatial data, using silhouette
score analysis to determine optimal cluster count
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("outbreak_data_imputed.csv")

# -----------------------------
# Engineer spatial + severity features for clustering
# -----------------------------
cluster_features = ["latitude", "longitude", "CFR", "fatalities", "confirmed_cases"]
X = df[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Try K = 2..12, evaluate with silhouette score
# -----------------------------
K_range = range(2, 13)
scores = []
models = {}

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    score = silhouette_score(X_scaled, km.labels_)
    scores.append(score)
    models[k] = km
    print(f"K={k:2d}  silhouette={score:.4f}")

best_k = list(K_range)[int(np.argmax(scores))]
print(f"\nOptimal K = {best_k} (silhouette = {max(scores):.4f})")

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), scores, marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Analysis — Geospatial KMeans")
plt.legend()
plt.tight_layout()
plt.savefig("silhouette_scores.png", dpi=150)
plt.close()

# -----------------------------
# Fit final model with best K, attach cluster labels
# -----------------------------
final_model = models[best_k]
df["cluster"] = final_model.labels_

df.to_csv("outbreak_data_clustered.csv", index=False)
print("Saved outbreak_data_clustered.csv and silhouette_scores.png")
