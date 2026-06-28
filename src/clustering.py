"""
clustering.py
=============
K-Means clustering + Elbow / Silhouette diagnostics + PCA 2-D visualization.

Why K-Means?
    - Scales linearly with sample size and dimensions.
    - Produces interpretable centroids that map directly to business personas.
    - Industry-standard baseline for customer segmentation.

Why Elbow + Silhouette together?
    - The Elbow Method finds where adding more clusters yields diminishing
      WCSS reduction (a *visual* heuristic).
    - The Silhouette Score (-1 to 1) measures cohesion vs separation
      (a *quantitative* validation).
    - Using both prevents over-/under-segmentation.

Why PCA for visualization?
    - Customer feature space is high-dimensional. PCA projects to 2-D
      while preserving as much variance as possible — perfect for
      visually verifying cluster separation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .visualization import PALETTE, save_fig, set_style


# ---------------------------------------------------------------------------
def elbow_and_silhouette(
    X: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """Compute WCSS (inertia) and Silhouette Score across k values."""
    print("\n=== ELBOW & SILHOUETTE ===")
    set_style()
    wcss, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        print(f"  k={k:>2}  WCSS={km.inertia_:>10,.1f}  Silhouette={sil:.4f}")

    # ---- Elbow plot ----
    plt.figure()
    plt.plot(list(k_range), wcss, "o-", color=PALETTE[0], linewidth=2, markersize=8)
    plt.title("Elbow Method — Optimal Cluster Count")
    plt.xlabel("Number of clusters (k)"); plt.ylabel("WCSS (inertia)")
    save_fig("23_elbow_method")

    # ---- Silhouette plot ----
    plt.figure()
    plt.plot(list(k_range), silhouettes, "o-", color=PALETTE[1], linewidth=2, markersize=8)
    plt.title("Silhouette Score by k")
    plt.xlabel("Number of clusters (k)"); plt.ylabel("Silhouette Score")
    save_fig("24_silhouette_scores")

    return {"k": list(k_range), "wcss": wcss, "silhouette": silhouettes}


# ---------------------------------------------------------------------------
def fit_kmeans(X: pd.DataFrame, k: int, random_state: int = 42) -> KMeans:
    """Fit K-Means with a fixed k."""
    print(f"\n=== FITTING K-MEANS (k={k}) ===")
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(X)
    return km


# ---------------------------------------------------------------------------
def visualize_pca(X: pd.DataFrame, labels: np.ndarray, k: int) -> Path:
    """Project to 2-D PCA and plot clusters."""
    print("\n=== PCA VISUALIZATION ===")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    plt.figure(figsize=(11, 7))
    palette = sns.color_palette(PALETTE, n_colors=k)
    for c in range(k):
        m = labels == c
        plt.scatter(coords[m, 0], coords[m, 1], s=30, alpha=0.75,
                    color=palette[c], label=f"Cluster {c}")
    plt.title(f"Customer Segments — PCA 2-D Projection  "
              f"(Var explained: {sum(var)*100:.1f}%)")
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.legend(loc="best", frameon=True)
    return save_fig("25_pca_clusters")


# ---------------------------------------------------------------------------
def cluster_profile_heatmap(profile: pd.DataFrame) -> Path:
    """Z-score heatmap of cluster centroids → quick persona-spotting."""
    z = (profile - profile.mean()) / profile.std()
    plt.figure(figsize=(12, max(4, 0.5 * len(profile.columns))))
    sns.heatmap(z.T, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                linewidths=0.4, cbar_kws={"shrink": 0.6})
    plt.title("Cluster Profile Heatmap (z-scored feature means)")
    plt.xlabel("Cluster"); plt.ylabel("Feature")
    return save_fig("26_cluster_profile_heatmap")


# ---------------------------------------------------------------------------
def cluster_size_donut(labels: np.ndarray) -> Path:
    """Donut chart of cluster sizes."""
    counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(8, 8))
    wedges, _, _ = plt.pie(
        counts.values,
        labels=[f"Cluster {i}\n{v:,}" for i, v in counts.items()],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        colors=PALETTE[: len(counts)],
    )
    plt.title("Customer Distribution Across Clusters")
    return save_fig("27_cluster_sizes_donut")


# ---------------------------------------------------------------------------
def cluster_revenue_bar(df: pd.DataFrame, label_col: str = "Cluster") -> Path:
    """Revenue contribution per cluster."""
    rev = df.groupby(label_col)["Total_Spend"].sum().sort_index()
    plt.figure()
    ax = sns.barplot(x=rev.index, y=rev.values, palette=PALETTE[: len(rev)])
    plt.title("Revenue Contribution by Cluster")
    plt.xlabel("Cluster"); plt.ylabel("Total Revenue")
    for p, v in zip(ax.patches, rev.values):
        ax.annotate(f"{int(v):,}", (p.get_x() + p.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    return save_fig("28_cluster_revenue")
