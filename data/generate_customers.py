from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

tenure_months = np.clip(rng.exponential(24, n).round(), 1, 72).astype(int)
monthly_charges = np.clip(rng.normal(70, 25, n), 20, 150).round(2)
support_tickets = rng.poisson(1.5, n)
late_payments = rng.poisson(0.8, n)
contract_type = rng.choice(["Monthly", "Annual", "Two-Year"], size=n, p=[0.55, 0.30, 0.15])
app_engagement = rng.beta(2, 2, n).round(3)

contract_effect = pd.Series(contract_type).map({"Monthly": 0.8, "Annual": -0.3, "Two-Year": -0.9}).to_numpy()
logit = (
    -1.0
    - 0.04 * tenure_months
    + 0.015 * (monthly_charges - 70)
    + 0.35 * support_tickets
    + 0.40 * late_payments
    - 1.5 * app_engagement
    + contract_effect
)
churn_probability = 1 / (1 + np.exp(-logit))
churned = (rng.random(n) < churn_probability).astype(int)

df = pd.DataFrame({
    "customer_id": np.arange(1, n + 1),
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "support_tickets": support_tickets,
    "late_payments": late_payments,
    "contract_type": contract_type,
    "app_engagement": app_engagement,
    "churned": churned,
})

out_path = Path(__file__).parent / "customers.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} customers to {out_path}")
print(f"Churn rate: {df['churned'].mean():.3f}")
