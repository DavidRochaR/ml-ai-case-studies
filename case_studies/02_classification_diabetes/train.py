"""Diabetes classification — training script."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate_classification, compare_models
from model import get_models


def main():
    # Load data (Pima Indians Diabetes — use load_diabetes as proxy or fetch openml)
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
        X, y = data.data, (data.target == "tested_positive").astype(int)
    except Exception:
        # Fallback synthetic
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=768, n_features=8, random_state=42)
        X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
        y = pd.Series(y)

    print(f"Dataset shape: {X.shape}, positive class: {y.sum()}/{len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    for name, model in get_models().items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None
        results.append(evaluate_classification(y_test, preds, probs, name=name))

    print("\n=== Model Comparison ===")
    print(compare_models(results).round(4))


if __name__ == "__main__":
    main()
