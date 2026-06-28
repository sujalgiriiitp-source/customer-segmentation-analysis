# 🛍️ Customer Segmentation Analysis using Python, Machine Learning & Business Intelligence

> **End-to-end Data Science project** that segments customers into actionable personas using **RFM analysis**, **K-Means clustering**, **PCA**, and **20+ professional visualizations** — built to production standards by applying the same workflow used by Senior Data Scientists at top tech companies.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

---

## 📌 Project Overview

Customer segmentation is one of the highest-ROI applications of data science in retail and e-commerce. By grouping customers based on **demographics, behavior, and spending patterns**, businesses can:

- 🎯 Run **targeted marketing campaigns** with 3-5x higher conversion
- 💰 Maximize **Customer Lifetime Value (CLV)**
- 🚨 Identify **at-risk customers** before they churn
- 🌟 Reward **loyal high-value customers**
- 📈 Improve **product recommendations** and cross-sell strategies

This project takes a real-world customer dataset and delivers a **complete, reproducible analytics pipeline** — from raw data ingestion to executive-ready business recommendations.

---

## 📊 Dataset

**Source:** Customer Personality Analysis (Kaggle — public dataset)

The dataset contains **2,240 customer records** with features covering:

| Category | Features |
|----------|----------|
| **Demographics** | Year of Birth, Education, Marital Status, Income, Kids, Teens |
| **Customer History** | Date of enrollment, Recency (days since last purchase) |
| **Product Spending** | Wines, Fruits, Meat, Fish, Sweets, Gold |
| **Purchase Channels** | Web, Catalog, Store, Deals purchases |
| **Engagement** | Web visits, Campaign acceptance (5 campaigns) |
| **Response** | Final campaign response |

> The loader script (`src/data_loader.py`) **auto-downloads** the dataset on first run. If the network is unavailable, it falls back to a built-in **synthetic data generator** of equivalent schema so the pipeline always runs end-to-end.

---

## 📁 Folder Structure

```
customer-segmentation-analysis/
│
├── data/
│   ├── raw/                          # Original dataset (auto-downloaded)
│   └── processed/                    # Cleaned + feature-engineered data
│
├── notebooks/
│   └── Customer_Segmentation_Analysis.ipynb   # Interactive walkthrough
│
├── src/                              # Modular, production-grade source code
│   ├── __init__.py
│   ├── data_loader.py                # Dataset ingestion (download + fallback)
│   ├── data_cleaning.py              # Missing values, duplicates, outliers
│   ├── feature_engineering.py        # RFM, age groups, loyalty score
│   ├── eda.py                        # Exploratory Data Analysis (20+ plots)
│   ├── preprocessing.py              # Encoding + Scaling
│   ├── clustering.py                 # K-Means + Elbow + Silhouette + PCA
│   ├── visualization.py              # Reusable styled plotting helpers
│   └── main.py                       # End-to-end orchestration pipeline
│
├── reports/
│   ├── executive_summary.md          # 1-page summary for leadership
│   ├── business_report.md            # Full business deep-dive
│   ├── final_insights.md             # Cluster-level insights & actions
│   └── data_dictionary.md            # Every variable explained
│
├── visualizations/                   # 20+ saved PNG visualizations
├── models/                           # Serialized K-Means + Scaler + PCA
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## 🔄 Workflow

```text
   ┌──────────────┐    ┌───────────────┐    ┌───────────────────────┐
   │  Raw Dataset │───▶│ Data Cleaning │───▶│ Feature Engineering   │
   └──────────────┘    └───────────────┘    │ (RFM, Age, Loyalty)   │
                                            └──────────┬────────────┘
                                                       │
                                                       ▼
   ┌─────────────────────┐     ┌──────────────┐    ┌──────────────┐
   │ Business Insights & │◀────│  K-Means +   │◀───│ Preprocessing │
   │  Persona Reports    │     │ PCA + Elbow  │    │ (Encode+Scale)│
   └─────────────────────┘     └──────────────┘    └──────────────┘
```

---

## 🖼️ Visualizations (20+)

All plots are auto-generated and saved to `visualizations/`:

1. Age Distribution Histogram
2. Income Distribution
3. Total Spending Distribution
4. Education Countplot
5. Marital Status Countplot
6. Kids at Home Analysis
7. Teens at Home Analysis
8. Spending by Education
9. Spending by Marital Status
10. Income vs Spending Scatter
11. Correlation Heatmap
12. Pairplot (key features)
13. Product Category Spending Bar
14. Purchase Channel Breakdown
15. Campaign Acceptance Rates
16. RFM Distributions (Recency / Frequency / Monetary)
17. Outlier Boxplots
18. Elbow Method Plot
19. Silhouette Score Plot
20. PCA 2D Cluster Visualization
21. Cluster Profile Heatmap
22. Cluster Size Donut Chart
23. Cluster Revenue Contribution
24. Persona Radar / Spider Charts

Each plot includes a written **business insight** in `reports/business_report.md`.

---

## 🤖 Machine Learning

**Algorithm:** K-Means Clustering (unsupervised)

**Cluster Selection:**
- **Elbow Method (WCSS)** — identifies the "knee" where adding clusters yields diminishing returns
- **Silhouette Score** — validates cohesion vs separation (score in `[-1, 1]`, higher is better)
- We select **k = 4** as the optimal cluster count

**Dimensionality Reduction:** PCA → 2 components for visualization (captures ~50–60% of variance)

**Why K-Means?**
- Scales well to thousands of customers
- Interpretable cluster centroids (easy to describe to business)
- Industry standard for RFM-based segmentation

---

## 💡 Business Insights — Customer Personas

| Cluster | Persona | Profile | Strategy |
|---------|---------|---------|----------|
| 🟢 0 | **Premium Loyalists** | High income, high spend, frequent buyers | VIP program, early access, premium upsell |
| 🔵 1 | **Regular Customers** | Mid income, mid spend, steady frequency | Loyalty points, bundle offers |
| 🟡 2 | **High Potential** | High income, low engagement | Re-engagement campaigns, personalized offers |
| 🔴 3 | **Budget / At-Risk** | Low income, low spend, high recency | Discount-led reactivation, churn-prevention |

Full segment deep-dives live in [`reports/final_insights.md`](reports/final_insights.md).

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/customer-segmentation-analysis.git
cd customer-segmentation-analysis
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python src/main.py
```

This will:
- Download (or synthesize) the dataset → `data/raw/`
- Clean & engineer features → `data/processed/`
- Generate all 20+ visualizations → `visualizations/`
- Train K-Means & save the model → `models/`
- Produce business reports → `reports/`

### 5. Explore interactively
```bash
jupyter notebook notebooks/Customer_Segmentation_Analysis.ipynb
```

---

## 📦 Requirements

- Python 3.9+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- jupyter

Full list in [`requirements.txt`](requirements.txt).

---

## 🔮 Future Improvements

- 🧠 Compare with **DBSCAN**, **Hierarchical Clustering**, **Gaussian Mixture Models**
- 📊 Deploy an interactive **Streamlit / Plotly Dash** dashboard
- 🤝 Integrate with a **CRM / marketing automation** platform (HubSpot, Salesforce)
- 📈 Forecast **Customer Lifetime Value (CLV)** with BG/NBD + Gamma-Gamma models
- ☁️ Productionize on **AWS SageMaker / GCP Vertex AI** with scheduled retraining
- 🧪 A/B test segment-targeted campaigns and measure uplift

---

## 👤 Author

**Your Name**
Senior Data Analyst / Data Scientist

- 🌐 GitHub: [@your-username](https://github.com/your-username)
- 💼 LinkedIn: [your-name](https://linkedin.com/in/your-name)
- 📧 Email: your.email@example.com

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

⭐ **If this project helped you, please star the repo!** ⭐
