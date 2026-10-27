from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
days = 365
t = np.arange(days)

baseline = 100
trend = 0.10 * t
weekly_pattern = 15 * np.sin(2 * np.pi * t / 7)
noise = rng.normal(0, 5, days)

orders = np.clip(baseline + trend + weekly_pattern + noise, 0, None).round().astype(int)

df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=days, freq="D"),
    "orders": orders,
})

out_path = Path(__file__).parent / "orders.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} days of orders to {out_path}")
print(f"Average daily orders: {df['orders'].mean():.1f}")
