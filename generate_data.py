import pandas as pd
import numpy as np
from datetime import datetime, timedelta
np.random.seed(99)

n_customers = 2000
n_orders    = 15000
snapshot    = datetime(2024, 12, 31)

customer_ids = [f'CUST-{str(i).zfill(4)}' for i in range(1, n_customers+1)]
segments_true = np.random.choice(
    ['Champions','Loyal','At Risk','Lost','New'],
    n_customers, p=[0.15,0.25,0.20,0.20,0.20])

rows = []
for i, cid in enumerate(customer_ids):
    seg = segments_true[i]
    if seg == 'Champions':
        n   = np.random.randint(12, 30)
        base_days_ago = np.random.randint(1, 15)
        base_value    = np.random.uniform(800, 3000)
    elif seg == 'Loyal':
        n   = np.random.randint(6, 15)
        base_days_ago = np.random.randint(10, 45)
        base_value    = np.random.uniform(400, 1200)
    elif seg == 'At Risk':
        n   = np.random.randint(3, 8)
        base_days_ago = np.random.randint(60, 150)
        base_value    = np.random.uniform(200, 800)
    elif seg == 'Lost':
        n   = np.random.randint(1, 4)
        base_days_ago = np.random.randint(180, 365)
        base_value    = np.random.uniform(50, 300)
    else:  # New
        n   = np.random.randint(1, 3)
        base_days_ago = np.random.randint(1, 30)
        base_value    = np.random.uniform(100, 500)

    categories = np.random.choice(
        ['Electronics','Clothing','Home & Kitchen','Books','Sports','Beauty'],
        n, p=[0.25,0.20,0.20,0.15,0.10,0.10])

    for j in range(n):
        days_ago = base_days_ago + j * np.random.randint(5, 25)
        order_date = snapshot - timedelta(days=int(days_ago))
        rows.append({
            'CustomerID':   cid,
            'OrderID':      f'ORD-{np.random.randint(100000,999999)}',
            'OrderDate':    order_date.strftime('%Y-%m-%d'),
            'Category':     categories[j],
            'Quantity':     np.random.randint(1, 6),
            'UnitPrice':    round(base_value / np.random.randint(1,5), 2),
            'TotalAmount':  round(base_value * np.random.uniform(0.7, 1.3), 2),
            'City':         np.random.choice(
                ['Mumbai','Pune','Bangalore','Delhi','Hyderabad','Chennai'],
                p=[0.22,0.18,0.20,0.15,0.13,0.12]),
            'PaymentMode':  np.random.choice(
                ['UPI','Credit Card','Debit Card','Net Banking','COD'],
                p=[0.35,0.25,0.20,0.12,0.08])
        })

df = pd.DataFrame(rows)
df.to_csv('/home/claude/projects/ecommerce_rfm/data/ecommerce_orders.csv', index=False)
print(f"Orders: {len(df):,} | Customers: {df['CustomerID'].nunique():,}")
print(f"Date range: {df['OrderDate'].min()} → {df['OrderDate'].max()}")
