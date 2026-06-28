# 🧭 Final Insights — Customer Segments & Action Plan

This document translates the K-Means clusters into **named personas** with concrete marketing, retention, and growth actions.

> Cluster ordering follows centroid ranking on income, spend, and loyalty (descending) minus recency.

---

## 🟢 Cluster A — Premium Loyalists

**Profile**
- Highest income & total spend
- Long customer tenure
- Low recency (recent buyers)
- Highest loyalty score
- Few children at home

**Why they matter:** ~20% of customers, ~50% of revenue.

**Recommendations**
1. **VIP / Tier-1 loyalty program** with early access, free shipping, exclusive bundles.
2. **Concierge service** — dedicated account manager via email or chatbot.
3. **Premium product launches** previewed to this segment first.
4. **Referral rewards** — these customers convert their network well.

**KPIs:** Retention rate, AOV, NPS.

---

## 🔵 Cluster B — Regular Customers

**Profile**
- Middle-income, middle-spend
- Consistent purchase frequency across channels
- Mixed household composition

**Why they matter:** The volume backbone of the business — small per-customer lift compounds to large absolute revenue.

**Recommendations**
1. **Tier-2 loyalty program** with points & milestones.
2. **Cross-sell bundles** anchored on Wine + Meat (top categories).
3. **Targeted email cadence** (1–2 sends/week with personalized product picks).
4. Promote **web purchases** with free-shipping thresholds to lift AOV.

**KPIs:** AOV, repeat-purchase rate, category cross-sell rate.

---

## 🟡 Cluster C — High Potential

**Profile**
- Above-average income
- Below-average spend & engagement
- Mid-to-high recency
- Often parents (kids reduce premium category spend)

**Why they matter:** Largest **wallet-share upside** in the entire base.

**Recommendations**
1. **Personalized re-engagement journey** — survey "what brings you back?" and tailor content.
2. **Family-oriented bundles** (snacks, sweets, gold gifting).
3. **Time-limited premium trial** (e.g., free sample of premium wine with next order).
4. **Behavioral retargeting** on web visits with no purchase.

**KPIs:** Wallet-share growth, conversion rate from web visit → purchase.

---

## 🔴 Cluster D — Budget / At-Risk

**Profile**
- Low income, low spend
- High recency (haven't purchased recently)
- Lower loyalty score
- Often respond to deal campaigns

**Why they matter:** Highest **churn risk** — but cheap to reactivate.

**Recommendations**
1. **Win-back campaign** — "We miss you" + 15–20% discount.
2. **Deal-led email** with the cheapest-entry products.
3. **Surveys** to understand churn drivers and act on top complaints.
4. **Suppress** from premium / full-price campaigns to preserve brand value.

**KPIs:** Reactivation rate, churn rate, cost-per-reactivation.

---

## 📅 Operating Cadence

| Cadence | Action |
|---------|--------|
| **Weekly** | Persona dashboards refresh, campaign performance review |
| **Monthly** | Segment migration analysis (who moved between clusters?) |
| **Quarterly** | Re-run full clustering pipeline; re-validate personas |
| **Annually** | Re-evaluate feature set, test alternative algorithms (DBSCAN, GMM, hierarchical) |

---

## 🧪 Recommended A/B Tests

1. VIP program enrollment offer (Cluster A): test perks vs cashback.
2. Family-bundle creative (Cluster C): test "save money" vs "save time" messaging.
3. Win-back discount depth (Cluster D): 10% vs 20% vs free shipping.
4. Email cadence (Cluster B): 1/week vs 2/week.

Measure with **incremental revenue** vs hold-out, not raw open/click rates.
