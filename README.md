# 🛒 E-Commerce Customer Segmentation — RFM Analysis

> **Which customers are your champions? Which are about to leave?**
>
> Built by **Baldev Rathod** · Data Analyst & ML Engineer · Pune

---

## 🎯 Problem Statement

Not all customers are equal. A clothing brand with 2,000 customers can't treat a first-time buyer the same as a loyal repeat customer spending ₹28,000. This project uses **RFM Analysis** (Recency, Frequency, Monetary) combined with **K-Means Clustering** to segment customers into actionable groups — enabling targeted marketing, retention, and growth strategies.

**Business question:** Which customers should we focus our marketing budget on, and how should each segment be treated differently?

---

## 📊 Dataset

| Property | Value |
|---|---|
| Orders | 14,089 transactions |
| Customers | 2,000 unique buyers |
| Date Range | March 2023 – December 2024 |
| Total GMV | ₹1.62 Crore |
| Categories | Electronics, Clothing, Home, Books, Sports, Beauty |
| Cities | Mumbai, Pune, Bangalore, Delhi, Hyderabad, Chennai |

---

## 🔍 RFM Framework

| Dimension | What It Measures | Business Signal |
|---|---|---|
| **Recency (R)** | Days since last purchase | How engaged is this customer today? |
| **Frequency (F)** | Number of orders | How loyal are they? |
| **Monetary (M)** | Total amount spent | How valuable are they? |

Each customer is scored 1–5 on each dimension. Combined score (3–15) maps to a segment.

---

## 📊 Segment Summary

| Segment | Customers | Avg Recency | Avg Orders | Avg Spend | Total Revenue |
|---|---|---|---|---|---|
| **Champions** | 445 | 11 days | 17 orders | ₹28,002 | ₹1.25 Cr |
| **Loyal Customers** | 338 | 35 days | 9 orders | ₹7,229 | ₹24.4 L |
| **At Risk** | 340 | 107 days | 5 orders | ₹2,659 | ₹9.0 L |
| **New Customers** | 336 | 12 days | 1.5 orders | ₹456 | ₹1.5 L |
| **Need Attention** | 212 | 130 days | 2.4 orders | ₹690 | ₹1.5 L |
| **Lost** | 329 | 278 days | 1.7 orders | ₹306 | ₹1.0 L |

---

## 🧠 Approach

```
Transactions → RFM Calculation → Scoring (1–5) → Segmentation → K-Means Validation → Recommendations
```

**K-Means Clustering** was used to validate RFM segments:
- Best k = 3 (Silhouette score = 0.59)
- Confirms 3 broad behavioural groups: High Value, Mid Value, Low/Inactive

---

## 💡 Business Recommendations

| Segment | Strategy |
|---|---|
| **Champions** | Reward with loyalty points, early access to sales, referral programs |
| **Loyal Customers** | Upsell premium categories (Electronics), subscription offers |
| **At Risk** | Win-back email: "We miss you — here's 20% off" |
| **New Customers** | Onboarding series: second purchase discount within 14 days |
| **Lost** | Last-chance reactivation + survey to understand why they left |

**Revenue opportunity:** Retaining 30% of At-Risk customers = ₹2.71 Lakh additional revenue

---

## 🗂️ Project Structure

```
ecommerce_rfm/
├── data/
│   └── ecommerce_orders.csv         # 14,089 order transactions
├── src/
│   ├── generate_data.py             # Dataset generation
│   └── rfm_analysis.py              # Full RFM + clustering pipeline
├── outputs/
│   ├── 01_rfm_eda.png               # EDA dashboard (6 charts)
│   ├── 02_segment_deepdive.png      # Segment analysis charts
│   ├── rfm_customer_segments.csv    # Customer-level RFM scores + segments
│   └── segment_summary.csv         # Segment-level business summary
└── README.md
```

---

## 🛠️ Tech Stack

`Python 3.11` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

---

## 🚀 How to Run

```bash
git clone https://github.com/yourgithub/ecommerce-rfm-segmentation
cd ecommerce-rfm-segmentation
pip install pandas numpy scikit-learn matplotlib seaborn
python3 src/generate_data.py
python3 src/rfm_analysis.py
```

---

## 📬 Contact

**Baldev Rathod** | okbaldev123@gmail.com | [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourgithub)
