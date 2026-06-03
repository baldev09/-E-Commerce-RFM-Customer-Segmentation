# =============================================================
# E-Commerce Customer Segmentation using RFM Analysis
# Author : Baldev Rathod  |  GitHub: github.com/yourgithub
# Dataset: 14,089 orders | 2,000 customers | Indian e-commerce
# Goal   : Segment customers by behaviour → targeted marketing
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.titleweight':'bold','figure.dpi':120})
PALETTE  = ['#2563EB','#10B981','#F59E0B','#EF4444','#8B5CF6','#EC4899']
OUTPUT   = '/home/claude/projects/ecommerce_rfm/outputs'
SNAPSHOT = datetime(2024, 12, 31)

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  E-COMMERCE RFM SEGMENTATION — Baldev Rathod")
print("="*60)

df = pd.read_csv('/home/claude/projects/ecommerce_rfm/data/ecommerce_orders.csv',
                 parse_dates=['OrderDate'])

print(f"\n[DATA]  Orders    : {len(df):,}")
print(f"[DATA]  Customers : {df['CustomerID'].nunique():,}")
print(f"[DATA]  Date range: {df['OrderDate'].min().date()} → {df['OrderDate'].max().date()}")
print(f"[DATA]  Nulls     : {df.isnull().sum().sum()}")
print(f"[DATA]  Total GMV : ₹{df['TotalAmount'].sum():,.0f}")

# ─────────────────────────────────────────────
# 2. COMPUTE RFM METRICS
# ─────────────────────────────────────────────
rfm = df.groupby('CustomerID').agg(
    Recency    = ('OrderDate',   lambda x: (SNAPSHOT - x.max()).days),
    Frequency  = ('OrderID',     'nunique'),
    Monetary   = ('TotalAmount', 'sum')
).reset_index()
rfm['Monetary'] = rfm['Monetary'].round(2)

print(f"\n[RFM]   Median Recency  : {rfm['Recency'].median():.0f} days")
print(f"[RFM]   Median Frequency: {rfm['Frequency'].median():.0f} orders")
print(f"[RFM]   Median Monetary : ₹{rfm['Monetary'].median():,.0f}")

# ─────────────────────────────────────────────
# 3. RFM SCORING (1–5)
# ─────────────────────────────────────────────
rfm['R_Score'] = pd.qcut(rfm['Recency'],   5, labels=[5,4,3,2,1]).astype(int)  # lower recency = better
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'),  5, labels=[1,2,3,4,5]).astype(int)
rfm['RFM_Score']   = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

def label_segment(row):
    r,f,m = row['R_Score'], row['F_Score'], row['M_Score']
    score = row['RFM_Score']
    if r>=4 and f>=4 and m>=4:    return 'Champions'
    elif r>=3 and f>=3 and m>=3:  return 'Loyal Customers'
    elif r>=4 and f<=2:           return 'New Customers'
    elif r<=2 and f>=3 and m>=3:  return 'At Risk'
    elif r<=2 and f>=4:           return 'Cannot Lose Them'
    elif r<=1 and f<=2:           return 'Lost'
    elif score >= 9:              return 'Potential Loyalists'
    else:                         return 'Need Attention'

rfm['Segment'] = rfm.apply(label_segment, axis=1)

seg_summary = (rfm.groupby('Segment')
                  .agg(Count=('CustomerID','count'),
                       Avg_Recency=('Recency','mean'),
                       Avg_Frequency=('Frequency','mean'),
                       Avg_Monetary=('Monetary','mean'),
                       Total_Revenue=('Monetary','sum'))
                  .round(1)
                  .sort_values('Total_Revenue', ascending=False))

print("\n[SEGMENTS]")
print(seg_summary.to_string())

# ─────────────────────────────────────────────
# 4. K-MEANS CLUSTERING (validation)
# ─────────────────────────────────────────────
scaler  = StandardScaler()
X_rfm   = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

sil_scores = {}
for k in range(3,8):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_rfm)
    sil_scores[k] = silhouette_score(X_rfm, lbl)

best_k   = max(sil_scores, key=sil_scores.get)
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['KMeans_Cluster'] = km_final.fit_predict(X_rfm)
print(f"\n[KMEANS] Best k={best_k}, Silhouette={sil_scores[best_k]:.4f}")

# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
seg_colors = {
    'Champions':'#10B981','Loyal Customers':'#2563EB','New Customers':'#8B5CF6',
    'At Risk':'#F59E0B','Cannot Lose Them':'#EF4444','Lost':'#6B7280',
    'Potential Loyalists':'#EC4899','Need Attention':'#0EA5E9'
}

# Chart 1 — EDA overview
fig, axes = plt.subplots(2, 3, figsize=(18,11))
fig.suptitle('E-Commerce Customer Segmentation — RFM Analysis', fontsize=15)

# Segment counts
seg_cnt = rfm['Segment'].value_counts()
colors_bar = [seg_colors.get(s,'#aaa') for s in seg_cnt.index]
axes[0,0].barh(seg_cnt.index, seg_cnt.values, color=colors_bar, edgecolor='white')
axes[0,0].set_title('Customer Count by Segment')
axes[0,0].set_xlabel('Customers')
for i,v in enumerate(seg_cnt.values):
    axes[0,0].text(v+3, i, str(v), va='center', fontsize=9)

# Revenue by segment
seg_rev = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
colors_rev = [seg_colors.get(s,'#aaa') for s in seg_rev.index]
axes[0,1].barh(seg_rev.index, seg_rev.values/1e6, color=colors_rev, edgecolor='white')
axes[0,1].set_title('Total Revenue by Segment (₹M)')
axes[0,1].set_xlabel('Revenue (₹ Millions)')

# Recency distribution
rfm['Recency'].hist(bins=30, color=PALETTE[0], alpha=0.8, ax=axes[0,2], edgecolor='white')
axes[0,2].set_title('Recency Distribution (days since last order)')
axes[0,2].set_xlabel('Days')
axes[0,2].set_ylabel('Customers')

# Frequency distribution (capped at 20)
rfm['Frequency'].clip(upper=20).hist(bins=20, color=PALETTE[1], alpha=0.8, ax=axes[1,0], edgecolor='white')
axes[1,0].set_title('Frequency Distribution (orders per customer)')
axes[1,0].set_xlabel('Number of Orders')
axes[1,0].set_ylabel('Customers')

# Monetary distribution (log scale)
rfm['Monetary'].hist(bins=30, color=PALETTE[2], alpha=0.8, ax=axes[1,1], edgecolor='white')
axes[1,1].set_title('Monetary Distribution (₹ total spend)')
axes[1,1].set_xlabel('Total Spend (₹)')
axes[1,1].set_ylabel('Customers')

# RFM Score Heatmap (R vs F coloured by M)
pivot = rfm.groupby(['R_Score','F_Score'])['Monetary'].mean().unstack()
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1,2],
            cbar_kws={'label':'Avg Monetary (₹)'})
axes[1,2].set_title('RFM Heatmap: Avg Spend by R×F Score')
axes[1,2].set_xlabel('Frequency Score')
axes[1,2].set_ylabel('Recency Score')

plt.tight_layout()
plt.savefig(f'{OUTPUT}/01_rfm_eda.png', bbox_inches='tight')
plt.close()

# Chart 2 — Segment deep-dive
fig2, axes2 = plt.subplots(1, 3, figsize=(18,6))
fig2.suptitle('RFM Segment Deep-Dive — Business Insights', fontsize=14)

# Avg metrics per segment
metrics = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1)
metrics_norm = (metrics - metrics.min()) / (metrics.max() - metrics.min())
sns.heatmap(metrics_norm.T, annot=metrics.T, fmt='.1f', cmap='RdYlGn',
            ax=axes2[0], cbar=False, linewidths=0.5)
axes2[0].set_title('Segment Avg Metrics\n(colour = normalised)')
axes2[0].set_ylabel('')
plt.setp(axes2[0].get_xticklabels(), rotation=30, ha='right', fontsize=8)

# Scatter R vs M
for seg, grp in rfm.groupby('Segment'):
    axes2[1].scatter(grp['Recency'], grp['Monetary'],
                     alpha=0.35, s=15, label=seg,
                     color=seg_colors.get(seg,'#aaa'))
axes2[1].set_xlabel('Recency (days)')
axes2[1].set_ylabel('Monetary (₹ total spend)')
axes2[1].set_title('Recency vs Monetary by Segment')
axes2[1].legend(fontsize=7, markerscale=2)

# K-Means silhouette vs k
axes2[2].plot(list(sil_scores.keys()), list(sil_scores.values()),
              marker='o', color=PALETTE[0], linewidth=2, markersize=8)
axes2[2].axvline(x=best_k, color='#EF4444',
                 linestyle='--', alpha=0.7, label=f'Best k={best_k}')
axes2[2].set_title(f'K-Means: Silhouette Scores\n(Best k={best_k})')
axes2[2].set_xlabel('Number of Clusters (k)')
axes2[2].set_ylabel('Silhouette Score')
axes2[2].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT}/02_segment_deepdive.png', bbox_inches='tight')
plt.close()

print(f"\n[PLOTS] Saved 2 charts to {OUTPUT}/")

# ─────────────────────────────────────────────
# 6. SAVE RESULTS
# ─────────────────────────────────────────────
rfm.to_csv(f'{OUTPUT}/rfm_customer_segments.csv', index=False)
seg_summary.to_csv(f'{OUTPUT}/segment_summary.csv')

print(f"\n[OUTPUT] rfm_customer_segments.csv and segment_summary.csv saved")

# category & city analysis
top_cat  = df.groupby('Category')['TotalAmount'].sum().sort_values(ascending=False)
top_city = df.groupby('City')['TotalAmount'].sum().sort_values(ascending=False)

print("\n[INSIGHTS] Revenue by Category:")
for cat, val in top_cat.items():
    print(f"  {cat:<20}: ₹{val:,.0f}")

print("\n[INSIGHTS] Revenue by City:")
for city, val in top_city.items():
    print(f"  {city:<15}: ₹{val:,.0f}")

print("\n" + "="*60)
print("  BUSINESS IMPACT SUMMARY")
print("="*60)
champ = rfm[rfm['Segment']=='Champions']
at_risk = rfm[rfm['Segment']=='At Risk']
print(f"  Champions ({len(champ)} customers) avg spend : ₹{champ['Monetary'].mean():,.0f}")
print(f"  At-Risk   ({len(at_risk)} customers) avg spend : ₹{at_risk['Monetary'].mean():,.0f}")
print(f"  If 30% At-Risk retained, revenue gain : ₹{at_risk['Monetary'].sum()*0.30:,.0f}")
print(f"  Total GMV segmented : ₹{rfm['Monetary'].sum():,.0f}")
print("="*60)
print("[DONE]  All outputs saved.\n")
