from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts as files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.ml.churn_data import split_churn_data
from src.ml.churn_models import build_logistic_regression, build_random_forest

CHART_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "evaluation"
CHART_DIR.mkdir(parents=True, exist_ok=True)

X_train, X_test, y_train, y_test = split_churn_data()

models = {
    "Logistic regression": build_logistic_regression(),
    "Random forest": build_random_forest(),
}
for model in models.values():
    model.fit(X_train, y_train)

# ---------- Part 1: five metrics for each model ----------
rows = []
for name, model in models.items():
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # the chance each customer leaves
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    })

results = pd.DataFrame(rows).set_index("model").round(3)
print(f"Baseline accuracy (always predict 'stays'): {1 - y_test.mean():.3f}")
print()
print(results)
print()

# ---------- Part 2: confusion matrices ----------
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, model) in zip(axes, models.items()):
    matrix = confusion_matrix(y_test, model.predict(X_test))
    tn, fp, fn, tp = matrix.ravel()
    print(f"{name}: caught {tp} leavers, missed {fn}, false alarms {fp}, correctly left alone {tn}")
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Stays", "Leaves"])
    ax.set_yticks([0, 1], ["Stays", "Leaves"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=14)
plt.tight_layout()
plt.savefig(CHART_DIR / "1_confusion_matrices.png")
plt.close()
print()

# ---------- Part 3: ROC curves ----------
plt.figure(figsize=(5, 4))
for name, model in models.items():
    false_pos, true_pos, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(false_pos, true_pos, label=name)
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing")
plt.xlabel("False alarm rate")
plt.ylabel("Share of leavers caught")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "2_roc_curves.png")
plt.close()

# ---------- Part 4: which features mattered to the forest ----------
importances = pd.Series(
    models["Random forest"].feature_importances_, index=X_train.columns
).sort_values()
plt.figure(figsize=(7, 4))
plt.barh(importances.index, importances.values, color="steelblue")
plt.title("Random Forest: Feature Importance")
plt.tight_layout()
plt.savefig(CHART_DIR / "3_feature_importance.png")
plt.close()

# ---------- Part 5: what happens when we change the decision threshold ----------
probabilities = models["Logistic regression"].predict_proba(X_test)[:, 1]
print("LOGISTIC REGRESSION: changing the threshold for 'will leave'")
print("threshold | precision | recall | f1    | customers flagged")
print("----------|-----------|--------|-------|------------------")
for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
    flagged = (probabilities >= threshold).astype(int)
    print(
        f"{threshold:>9.1f} | {precision_score(y_test, flagged):>9.3f} | "
        f"{recall_score(y_test, flagged):>6.3f} | {f1_score(y_test, flagged):.3f} | {flagged.sum():>16}"
    )

print()
print(f"Saved 3 charts to {CHART_DIR}")
