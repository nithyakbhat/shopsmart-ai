from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "docs" / "eda"
CHART_DIR.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(DATA_DIR / "customers.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["date"])

# ---------- Part 1: look at the data ----------
print("=== CUSTOMERS ===")
print("Shape:", customers.shape)
print(customers.head())
print()
print(customers.describe())
print()
print("Missing values per column:")
print(customers.isna().sum())
print()
print(f"Churn rate: {customers['churned'].mean():.3f}")

print()
print("=== ORDERS ===")
print("Shape:", orders.shape)
print(orders.head())
print()
print(orders["orders"].describe())

# ---------- Part 2: charts ----------
# Chart 1: how many customers churned vs stayed
counts = customers["churned"].value_counts().sort_index()
plt.figure(figsize=(5, 4))
plt.bar(["Retained", "Churned"], counts.values, color=["steelblue", "indianred"])
plt.title("Customers: Retained vs Churned")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig(CHART_DIR / "1_churn_distribution.png")
plt.close()

# Chart 2: churn rate by contract type
by_contract = customers.groupby("contract_type")["churned"].mean().sort_values(ascending=False)
plt.figure(figsize=(6, 4))
plt.bar(by_contract.index, by_contract.values, color="darkorange")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn rate")
plt.tight_layout()
plt.savefig(CHART_DIR / "2_churn_by_contract.png")
plt.close()

# Chart 3: tenure of churned vs retained customers
plt.figure(figsize=(7, 4))
plt.hist(customers.loc[customers["churned"] == 0, "tenure_months"], bins=30, alpha=0.6, label="Retained")
plt.hist(customers.loc[customers["churned"] == 1, "tenure_months"], bins=30, alpha=0.6, label="Churned")
plt.title("Tenure (months): Retained vs Churned")
plt.xlabel("Tenure (months)")
plt.ylabel("Number of customers")
plt.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "3_tenure_by_churn.png")
plt.close()

# Chart 4: churn rate by number of support tickets
tickets = customers.copy()
tickets["support_tickets"] = tickets["support_tickets"].clip(upper=6)  # group 6 or more together
by_tickets = tickets.groupby("support_tickets")["churned"].mean()
plt.figure(figsize=(6, 4))
plt.plot(by_tickets.index, by_tickets.values, marker="o")
plt.title("Churn Rate by Support Tickets (6 = 6 or more)")
plt.xlabel("Support tickets")
plt.ylabel("Churn rate")
plt.tight_layout()
plt.savefig(CHART_DIR / "4_churn_by_tickets.png")
plt.close()

# Chart 5: daily orders with a 7-day rolling average
orders["rolling_7d"] = orders["orders"].rolling(7).mean()
plt.figure(figsize=(10, 4))
plt.plot(orders["date"], orders["orders"], alpha=0.5, label="Daily orders")
plt.plot(orders["date"], orders["rolling_7d"], color="black", label="7-day average")
plt.title("Daily Orders Over Time")
plt.xlabel("Date")
plt.ylabel("Orders")
plt.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "5_daily_orders.png")
plt.close()

print()
print(f"Saved 5 charts to {CHART_DIR}")
