# Geospatial ML for Epidemic Forecasting (Reconstructed)

Reconstructed pipeline matching your resume bullets:
- Preprocessed outbreak data, applied **KNN imputation** to estimate missing
  fatalities using geolocation and CFR
- Applied **KMeans clustering** on geospatial data, using **silhouette score
  analysis** to determine optimal cluster count
- Engineered spatial features and computed **cluster-wise death and CFR
  statistics** to enhance predictive accuracy



## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python 00_generate_data.py           # builds outbreak_data.csv (skip if you have real data)
python 01_imputation.py              # KNN imputation -> outbreak_data_imputed.csv
python 02_clustering.py              # KMeans + silhouette scores -> outbreak_data_clustered.csv
python 03_cluster_stats_and_viz.py   # cluster-wise stats + plots
```

## Outputs

- `outbreak_data.csv` — raw synthetic data with missing fatalities
- `outbreak_data_imputed.csv` — after KNN imputation
- `outbreak_data_clustered.csv` — after KMeans clustering (adds `cluster` column)
- `cluster_statistics.csv` — cluster-wise death/CFR statistics
- `silhouette_scores.png` — silhouette score vs. K plot
- `cluster_map.png` — regions plotted geographically, colored by cluster, sized by CFR
- `cluster_cfr_barchart.png` — CFR comparison across clusters

## Notes

- KNN imputation uses `latitude`, `longitude`, `confirmed_cases`,
  `fatalities`, and `CFR` (scaled) as the feature space — the missing
  `fatalities`/`CFR` values are filled based on the 5 nearest regions in
  that space, weighted by distance.
- Clustering features combine geolocation with severity (`CFR`,
  `fatalities`, `confirmed_cases`) so clusters reflect both *where* regions
  are and *how* severely they were hit — matching "geospatial ML," not pure
  geographic clustering.
- Silhouette scores are computed for K = 2..12; the script auto-selects the
  best K. Your real project may have used a different K — check
  `silhouette_scores.png` and adjust `K_range` in `02_clustering.py` if needed.
