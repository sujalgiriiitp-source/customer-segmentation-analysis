# 📑 Data Dictionary

Authoritative reference for every column in the dataset and every engineered feature.

## Raw columns

| Column | Type | Description |
|--------|------|-------------|
| ID | int | Unique customer identifier |
| Year_Birth | int | Customer's year of birth |
| Education | category | Education level (consolidated: Undergraduate / Graduate / Postgraduate) |
| Marital_Status | category | Marital status (consolidated: Partner / Single) |
| Income | float | Annual household income |
| Kidhome | int | Number of young children at home |
| Teenhome | int | Number of teens at home |
| Dt_Customer | datetime | Date of enrollment with the company |
| Recency | int | Days since last purchase |
| MntWines | int | Spending on wine (last 2 years) |
| MntFruits | int | Spending on fruits |
| MntMeatProducts | int | Spending on meat products |
| MntFishProducts | int | Spending on fish |
| MntSweetProducts | int | Spending on sweets |
| MntGoldProds | int | Spending on gold products |
| NumDealsPurchases | int | Purchases made with a discount |
| NumWebPurchases | int | Purchases through the company's website |
| NumCatalogPurchases | int | Purchases through catalog |
| NumStorePurchases | int | Purchases in stores |
| NumWebVisitsMonth | int | Website visits in the last month |
| AcceptedCmp1..5 | int (0/1) | Accepted offer in each of 5 historic campaigns |
| Complain | int (0/1) | 1 if customer complained in the last 2 years |
| Response | int (0/1) | 1 if accepted the offer in the last campaign |
| Z_CostContact | int | Constant — dropped during cleaning |
| Z_Revenue | int | Constant — dropped during cleaning |

## Engineered features

| Feature | Formula | Purpose |
|---------|---------|---------|
| Age | reference_year − Year_Birth | Demographics |
| Age_Group | bins: 18-30 / 31-45 / 46-60 / 60+ | Demographics |
| Customer_Tenure_Days | reference_date − Dt_Customer | Loyalty |
| Children | Kidhome + Teenhome | Household |
| Family_Size | (Partner ? 2 : 1) + Children | Household |
| Is_Parent | Children > 0 | Behavioral |
| Income_Group | bins: Low / Mid / High / Premium | Segmentation |
| Total_Spend | Σ Mnt* columns | Monetary (RFM) |
| Total_Purchases | Σ Num*Purchases | Frequency (RFM) |
| Avg_Spend | Total_Spend / Total_Purchases | Behavior |
| Total_Accepted_Cmp | Σ AcceptedCmp1..5 | Marketing responsiveness |
| Loyalty_Score | 0.4·freq + 0.3·tenure + 0.3·(1-recency), scaled 0-100 | Composite loyalty |
| Frequency | = Total_Purchases | RFM |
| Monetary | = Total_Spend | RFM |
| Cluster | K-Means output | Segment ID |
| Persona | Mapped name | Human-readable segment |
