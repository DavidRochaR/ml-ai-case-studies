"""Housing price regression — training script."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate_regression, compare_models
from model import get_models


def main():
    # Load data
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    print(f"Dataset shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train and evaluate
    results = []
    for name, model in get_models().items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        results.append(evaluate_regression(y_test, preds, name=name))

    print("\n=== Model Comparison ===")
    print(compare_models(results).round(4))


if __name__ == "__main__":
    main()
