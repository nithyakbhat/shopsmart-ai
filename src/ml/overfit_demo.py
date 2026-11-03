from sklearn.ensemble import RandomForestClassifier

from src.ml.churn_data import split_churn_data

X_train, X_test, y_train, y_test = split_churn_data()

print("max_depth | train accuracy | test accuracy | gap")
print("----------|----------------|---------------|------")
for depth in [1, 3, 6, 12, None]:
    forest = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42)
    forest.fit(X_train, y_train)
    train_acc = forest.score(X_train, y_train)
    test_acc = forest.score(X_test, y_test)
    label = "None" if depth is None else str(depth)
    print(f"{label:>9} | {train_acc:>14.3f} | {test_acc:>13.3f} | {train_acc - test_acc:.3f}")
