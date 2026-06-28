"""
main.py
=======
End-to-end orchestration pipeline for the Customer Segmentation project.

Run with:
    python -m src.main      (recommended — relative imports work)
    or
    python src/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow `python src/main.py` execution
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.clustering import (
        cluster_profile_heatmap, cluster_revenue_bar, cluster_size_donut,
        elbow_and_silhouette, fit_kmeans, visualize_pca,
    )
    from src.data_cleaning import clean_data
    from src.data_loader import load_dataset
    from src.eda import run_eda
    from src.feature_engineering import engineer_features
    from src.preprocessing import build_feature_matrix
else:
    from .clustering import (
        cluster_profile_heatmap, cluster_revenue_bar, cluster_size_donut,
        elbow_and_silhouette, fit_kmeans, visualize_pca,
    )
    from .data_cleaning import clean_data
    from .data_loader import load_dataset
    from .eda import run_eda
    from .feature_engineering import engineer_features
    from .preprocessing import build_feature_matrix


# ---------------------------------------------------------------------------
PERSONA_TEMPLATE = {
    "Premium Loyalists":
        "High income, high spend, frequent buyers, long tenure, low recency.",
    "Regular Customers":
        "Mid income, steady spend & frequency. Reliable backbone of revenue.",
    "High Potential":
        "Above-average income but low engagement / spend — upside to unlock.",
    "Budget / At-Risk":
        "Low income, low spend, high recency — churn risk; win-back priority.",
}


def assign_personas(df_with_clusters: pd.DataFrame) -> dict[int, str]:
    """Map cluster IDs → persona names using centroid heuristics."""
    profile = df_with_clusters.groupby("Cluster")[
        ["Income", "Total_Spend", "Recency", "Total_Purchases", "Loyalty_Score"]
    ].mean()

    # Rank clusters
    score = (
        profile["Income"].rank()
        + profile["Total_Spend"].rank()
        + profile["Loyalty_Score"].rank()
        - profile["Recency"].rank()
    )
    ranked = score.sort_values(ascending=False).index.tolist()

    names = ["Premium Loyalists", "Regular Customers",
             "High Potential", "Budget / At-Risk"]
    return {cid: names[i] for i, cid in enumerate(ranked[: len(names)])}


# ---------------------------------------------------------------------------
def main() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    # 1) Load
    df_raw = load_dataset()

    # 2) Clean
    df_clean = clean_data(df_raw)
    df_clean.to_csv("data/processed/cleaned.csv", index=False)

    # 3) Feature engineering
    df_feat = engineer_features(df_clean)
    df_feat.to_csv("data/processed/engineered.csv", index=False)

    # 4) EDA — 22 plots
    run_eda(df_feat)

    # 5) Preprocess
    X_scaled, scaler = build_feature_matrix(df_feat)

    # 6) Elbow + Silhouette diagnostics
    diag = elbow_and_silhouette(X_scaled, k_range=range(2, 9))

    # Choose k automatically: max silhouette in 3..6
    candidate_ks = [k for k in diag["k"] if 3 <= k <= 6]
    cand_scores = [diag["silhouette"][diag["k"].index(k)] for k in candidate_ks]
    optimal_k = candidate_ks[int(np.argmax(cand_scores))]
    print(f"\n[main] Optimal k chosen by silhouette in [3,6]: {optimal_k}")

    # 7) Fit final model
    km = fit_kmeans(X_scaled, k=optimal_k)
    df_feat["Cluster"] = km.labels_

    # 8) PCA visualization
    visualize_pca(X_scaled, km.labels_, k=optimal_k)

    # 9) Profile + extra cluster plots
    profile_cols = [
        "Income", "Age", "Total_Spend", "Total_Purchases", "Recency",
        "Loyalty_Score", "Children", "Customer_Tenure_Days",
        "Total_Accepted_Cmp",
    ]
    profile = df_feat.groupby("Cluster")[profile_cols].mean().round(2)
    profile.to_csv("data/processed/cluster_profile.csv")
    cluster_profile_heatmap(profile)
    cluster_size_donut(km.labels_)
    cluster_revenue_bar(df_feat)

    # 10) Personas
    personas = assign_personas(df_feat)
    df_feat["Persona"] = df_feat["Cluster"].map(personas)
    df_feat.to_csv("data/processed/segmented_customers.csv", index=False)

    # 11) Save models
    joblib.dump(km, "models/kmeans.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    # 12) Print summary
    print("\n=== FINAL CLUSTER PROFILE ===")
    print(profile.to_string())
    print("\n=== PERSONA MAPPING ===")
    for cid, name in personas.items():
        print(f"  Cluster {cid}  ->  {name}")
        print(f"      {PERSONA_TEMPLATE[name]}")

    print("\n✅ Pipeline complete. See /visualizations, /reports, /models.")


if __name__ == "__main__":
    main()
