"""Customer segmentation — training script."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

from src.evaluation import evaluate_clustering, compare_models
from model import get_clusterers, reduce_dimensions


def main():
    # Synthetic customer data (4 segments)
    X, _ = make_blobs(n_samples=400, centers=4, n_features=5,
                       cluster_std=1.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    print(f"Dataset shape: {X.shape}")

    results = []
    for name, clusterer in get_clusterers(n_clusters=4).items():
        labels = clusterer.fit_predict(X)
        if len(set(labels)) > 1:
            results.append(evaluate_clustering(X, labels, name=name))

    print("\n=== Clustering Comparison ===")
    print(compare_models(results).round(4))

    # Project to 2D for visualization
    X_pca = reduce_dimensions(X, n_components=2)
    print(f"\nPCA explained variance available — see notebook for plots")


if __name__ == "__main__":
    main()
