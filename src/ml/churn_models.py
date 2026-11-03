from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression():
    """Scaling and logistic regression chained together as one model."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def build_random_forest(max_depth=6):
    """A forest of 200 decision trees. max_depth limits how complex each tree can get."""
    return RandomForestClassifier(n_estimators=200, max_depth=max_depth, random_state=42)
