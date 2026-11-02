from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "customers.csv"
TARGET = "churned"


def load_churn_data():
    """Load the customers file and return the features (X) and the label (y)."""
    customers = pd.read_csv(DATA_PATH)

    y = customers[TARGET]
    X = customers.drop(columns=["customer_id", TARGET])  # the ID is just a label, not a clue

    # Models need numbers, so turn contract_type into one 0/1 column per contract type
    X = pd.get_dummies(X, columns=["contract_type"], dtype=int)
    return X, y


def split_churn_data(test_size=0.25, random_state=42):
    """Split into training and test sets, keeping the same churn rate in both."""
    X, y = load_churn_data()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
