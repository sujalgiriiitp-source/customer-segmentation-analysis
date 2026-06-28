# 📘 Business Report — Customer Segmentation Analysis

## 1. Background & Objective

The company holds rich behavioral and demographic data on **2,240+ customers** but has historically marketed to its entire base with a one-size-fits-all approach, leaving significant revenue on the table. The objective of this project is to:

- Cluster customers into **actionable segments**
- Build **data-driven personas**
- Translate clusters into **specific marketing & retention actions**

## 2. Data

- **Source:** Customer Personality Analysis (public Kaggle dataset).
- **Rows:** ~2,240 customers
- **Columns:** 29 — demographics, household composition, 2-year product spend, purchase channels, campaign acceptance.
- **Quality issues addressed:** 24 missing incomes, a handful of extreme income outliers, two duplicate rows, two non-informative constant columns.

## 3. Methodology

1. **Cleaning** — drop constants, type conversion, dedupe, median/mode impute, IQR-based outlier capping.
2. **Feature engineering** — Age, Tenure, Total/Average Spend, Total Purchases, Children, Family Size, Income Group, Loyalty Score, RFM.
3. **EDA** — 22 visualizations covering distributions, relationships, channel mix, campaign performance and RFM.
4. **Preprocessing** — ordinal encoding (Education), one-hot encoding (Marital Status), `StandardScaler` for all numeric features.
5. **Clustering** — K-Means with Elbow (WCSS) + Silhouette diagnostics; optimal **k chosen automatically** by maximum silhouette in [3, 6].
6. **Visualization** — PCA 2-D projection of cluster separation.
7. **Persona mapping** — clusters ranked by income, spend, loyalty and recency; mapped to named personas.

## 4. Key Visual Insights

| # | Chart | Insight |
|---|-------|---------|
| 01 | Age Distribution | Customer base is mature (mode ~50). Marketing tone & channel must reflect this. |
| 02 | Income Distribution | Right-skewed with long tail — premium tier is small but lucrative. |
| 03 | Total Spend | Bi-modal: a large low-spend group + a smaller high-spend "whales" group. |
| 10 | Income vs Spend | Strong positive correlation; parents spend less at any given income. |
| 11 | Correlation Heatmap | Strongest links: Income↔Total Spend, MntWines↔NumStorePurchases. |
| 13 | Product Revenue | Wines & Meat dominate; bundle/anchor promotions on these. |
| 14 | Channel Mix | Store > Web > Catalog > Deals. Catalog is declining. |
| 15 | Campaign Acceptance | All historical campaigns ≤ 10% acceptance → segment-level targeting needed. |
| 16 | RFM | Highly skewed monetary; a small cohort drives most revenue. |
| 23 | Elbow Method | Curve bends at k ≈ 4. |
| 24 | Silhouette | Peak silhouette confirms 4 as the optimal cluster count. |
| 25 | PCA Clusters | Clusters separate cleanly in 2-D — segments are real, not random. |
| 26 | Cluster Profile Heatmap | Reveals each cluster's signature (income / spend / loyalty / recency). |
| 27 | Cluster Sizes Donut | Largest segments are mid-tier; premium is smallest but highest value. |
| 28 | Cluster Revenue | ~50% of revenue comes from ~20% of customers (Pareto confirmed). |

## 5. Segment Strategy

See `final_insights.md` for the full segment deep-dive and recommended playbook.

## 6. Risks & Limitations

- K-Means assumes roughly spherical clusters of similar size. Consider testing DBSCAN / GMM next iteration.
- Snapshot data — no time-series view of cluster migration yet.
- Behavior may shift post-promotion; segments should be **refreshed quarterly**.

## 7. Next Steps

1. Operationalize segment IDs into the **CRM** so marketing can target by persona.
2. Define **KPIs per segment** (retention, AOV, CLV).
3. A/B test persona-specific campaigns and measure uplift.
4. Build a **Streamlit dashboard** for marketing self-service.
