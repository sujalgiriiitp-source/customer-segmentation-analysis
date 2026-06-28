"""
eda.py
======
Exploratory Data Analysis — produces 20+ professional visualizations,
each saved to /visualizations as a PNG suitable for the business report.

Every function writes one chart and returns its file path so the pipeline
can collect them for the README / report.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .visualization import PALETTE, save_fig, set_style


# ---------------------------------------------------------------------------
def _annotate_bars(ax) -> None:
    """Add count labels above bar charts."""
    for p in ax.patches:
        h = p.get_height()
        if pd.notna(h) and h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9, color="#333")


# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> List[Path]:
    """Generate all EDA visualizations. Returns list of saved file paths."""
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    set_style()
    paths: List[Path] = []

    # ---- 1. Age distribution ----
    plt.figure()
    sns.histplot(df["Age"], bins=30, kde=True, color=PALETTE[0])
    plt.title("Customer Age Distribution")
    plt.xlabel("Age"); plt.ylabel("Customers")
    paths.append(save_fig("01_age_distribution"))

    # ---- 2. Income distribution ----
    plt.figure()
    sns.histplot(df["Income"], bins=40, kde=True, color=PALETTE[1])
    plt.title("Customer Income Distribution")
    plt.xlabel("Annual Income"); plt.ylabel("Customers")
    paths.append(save_fig("02_income_distribution"))

    # ---- 3. Total spending distribution ----
    plt.figure()
    sns.histplot(df["Total_Spend"], bins=40, kde=True, color=PALETTE[2])
    plt.title("Total Spending Distribution (Last 2 Years)")
    plt.xlabel("Total Spend"); plt.ylabel("Customers")
    paths.append(save_fig("03_total_spend_distribution"))

    # ---- 4. Education countplot ----
    plt.figure()
    ax = sns.countplot(data=df, x="Education", order=df["Education"].value_counts().index)
    _annotate_bars(ax)
    plt.title("Customers by Education Level")
    paths.append(save_fig("04_education_count"))

    # ---- 5. Marital Status countplot ----
    plt.figure()
    ax = sns.countplot(data=df, x="Marital_Status", order=df["Marital_Status"].value_counts().index)
    _annotate_bars(ax)
    plt.title("Customers by Marital Status")
    paths.append(save_fig("05_marital_status_count"))

    # ---- 6. Kids at home ----
    plt.figure()
    ax = sns.countplot(data=df, x="Kidhome")
    _annotate_bars(ax)
    plt.title("Number of Young Children at Home")
    paths.append(save_fig("06_kidhome_count"))

    # ---- 7. Teens at home ----
    plt.figure()
    ax = sns.countplot(data=df, x="Teenhome")
    _annotate_bars(ax)
    plt.title("Number of Teens at Home")
    paths.append(save_fig("07_teenhome_count"))

    # ---- 8. Spending by Education ----
    plt.figure()
    sns.boxplot(data=df, x="Education", y="Total_Spend",
                order=["Undergraduate", "Graduate", "Postgraduate"])
    plt.title("Total Spending by Education Level")
    paths.append(save_fig("08_spend_by_education"))

    # ---- 9. Spending by Marital Status ----
    plt.figure()
    sns.boxplot(data=df, x="Marital_Status", y="Total_Spend")
    plt.title("Total Spending by Marital Status")
    paths.append(save_fig("09_spend_by_marital"))

    # ---- 10. Income vs Spending scatter ----
    plt.figure()
    sns.scatterplot(data=df, x="Income", y="Total_Spend",
                    hue="Is_Parent", alpha=0.6, palette=[PALETTE[0], PALETTE[3]])
    plt.title("Income vs Total Spending  (colored by Parent)")
    paths.append(save_fig("10_income_vs_spend"))

    # ---- 11. Correlation heatmap ----
    num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ["ID", "Z_CostContact", "Z_Revenue"] if c in df.columns],
        errors="ignore",
    )
    plt.figure(figsize=(14, 10))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, annot=False,
                linewidths=0.3, cbar_kws={"shrink": 0.7})
    plt.title("Correlation Heatmap — Numeric Features")
    paths.append(save_fig("11_correlation_heatmap"))

    # ---- 12. Pairplot (key features) ----
    pair_cols = ["Income", "Total_Spend", "Total_Purchases", "Recency", "Age"]
    g = sns.pairplot(df[pair_cols].sample(min(600, len(df)), random_state=0),
                     diag_kind="kde", plot_kws={"alpha": 0.5, "s": 18})
    g.fig.suptitle("Pairplot — Key Customer Metrics", y=1.02)
    g.fig.savefig(Path("visualizations") / "12_pairplot.png", dpi=140, bbox_inches="tight")
    plt.close("all")
    paths.append(Path("visualizations") / "12_pairplot.png")

    # ---- 13. Product category spending ----
    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    totals = df[spend_cols].sum().sort_values(ascending=False)
    plt.figure()
    ax = sns.barplot(x=totals.index, y=totals.values, palette=PALETTE)
    plt.title("Revenue by Product Category")
    plt.xticks(rotation=30, ha="right"); plt.ylabel("Total Revenue")
    for p, v in zip(ax.patches, totals.values):
        ax.annotate(f"{int(v):,}", (p.get_x() + p.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    paths.append(save_fig("13_product_category_revenue"))

    # ---- 14. Purchase channel breakdown ----
    ch_cols = ["NumWebPurchases", "NumCatalogPurchases",
               "NumStorePurchases", "NumDealsPurchases"]
    channel_totals = df[ch_cols].sum()
    plt.figure()
    plt.pie(channel_totals, labels=[c.replace("Num", "").replace("Purchases", "")
                                    for c in ch_cols],
            autopct="%1.1f%%", colors=PALETTE, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    plt.title("Purchase Channel Mix")
    paths.append(save_fig("14_channel_mix"))

    # ---- 15. Campaign acceptance rates ----
    cmp_cols = [f"AcceptedCmp{i}" for i in range(1, 6)] + (["Response"] if "Response" in df else [])
    rates = (df[cmp_cols].mean() * 100).round(2)
    plt.figure()
    ax = sns.barplot(x=rates.index, y=rates.values, palette=PALETTE)
    plt.title("Marketing Campaign Acceptance Rate (%)")
    plt.ylabel("Acceptance %"); plt.xticks(rotation=30, ha="right")
    for p, v in zip(ax.patches, rates.values):
        ax.annotate(f"{v:.1f}%", (p.get_x() + p.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    paths.append(save_fig("15_campaign_rates"))

    # ---- 16. RFM distributions ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df["Recency"], bins=30, kde=True, color=PALETTE[0], ax=axes[0])
    axes[0].set_title("Recency (days)")
    sns.histplot(df["Frequency"], bins=30, kde=True, color=PALETTE[1], ax=axes[1])
    axes[1].set_title("Frequency (purchases)")
    sns.histplot(df["Monetary"], bins=30, kde=True, color=PALETTE[2], ax=axes[2])
    axes[2].set_title("Monetary (spend)")
    fig.suptitle("RFM — Recency / Frequency / Monetary", y=1.03, fontsize=16, weight="bold")
    fig.savefig(Path("visualizations") / "16_rfm_distributions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(Path("visualizations") / "16_rfm_distributions.png")

    # ---- 17. Outlier boxplots ----
    box_cols = ["Income", "Total_Spend", "Total_Purchases", "Recency", "Age"]
    plt.figure(figsize=(12, 6))
    df_box = df[box_cols].apply(lambda s: (s - s.mean()) / s.std())  # standardize for shared axis
    sns.boxplot(data=df_box, palette=PALETTE)
    plt.title("Outlier Check — Standardized Boxplots")
    plt.ylabel("Standardized Value")
    paths.append(save_fig("17_outlier_boxplots"))

    # ---- 18. Age Group vs Spending ----
    plt.figure()
    sns.barplot(data=df, x="Age_Group", y="Total_Spend",
                order=["18-30", "31-45", "46-60", "60+"],
                estimator=np.mean, palette=PALETTE, errorbar=None)
    plt.title("Average Total Spending by Age Group")
    paths.append(save_fig("18_spend_by_age_group"))

    # ---- 19. Income Group countplot ----
    plt.figure()
    ax = sns.countplot(data=df, x="Income_Group",
                       order=["Low", "Mid", "High", "Premium"])
    _annotate_bars(ax)
    plt.title("Customers by Income Group")
    paths.append(save_fig("19_income_group"))

    # ---- 20. Loyalty score distribution ----
    plt.figure()
    sns.histplot(df["Loyalty_Score"], bins=30, kde=True, color=PALETTE[5])
    plt.title("Loyalty Score Distribution")
    plt.xlabel("Loyalty Score (0–100)")
    paths.append(save_fig("20_loyalty_distribution"))

    # ---- 21. Children vs Spend ----
    plt.figure()
    sns.boxplot(data=df, x="Children", y="Total_Spend", palette=PALETTE)
    plt.title("Total Spending vs Number of Children")
    paths.append(save_fig("21_children_vs_spend"))

    # ---- 22. Tenure vs Loyalty ----
    plt.figure()
    sns.scatterplot(data=df, x="Customer_Tenure_Days", y="Loyalty_Score",
                    hue="Income_Group", alpha=0.6,
                    hue_order=["Low", "Mid", "High", "Premium"])
    plt.title("Customer Tenure vs Loyalty Score")
    paths.append(save_fig("22_tenure_vs_loyalty"))

    print(f"[eda] Saved {len(paths)} visualizations to /visualizations.")
    return paths
