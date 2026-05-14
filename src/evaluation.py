"""
Shared Evaluation Utilities
=============================

Reusable evaluation functions across regression, classification, and clustering tasks.

Author: David Rocha
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.metrics import (
    # Regression
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    # Classification
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    # Clustering
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)


def evaluate_regression(y_true, y_pred, name: str = "model") -> dict:
    """Comprehensive regression metrics."""
    return {
        "model": name,
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "n_samples": len(y_true),
    }


def evaluate_classification(y_true, y_pred, y_prob=None,
                              name: str = "model",
                              average: str = "binary") -> dict:
    """Comprehensive classification metrics."""
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "n_samples": len(y_true),
    }

    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                if y_prob.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def evaluate_clustering(X, labels, name: str = "model") -> dict:
    """Comprehensive clustering metrics (unsupervised)."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {"model": name, "error": "Need at least 2 clusters"}

    return {
        "model": name,
        "n_clusters": n_clusters,
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "n_samples": len(labels),
    }


def compare_models(results: List[dict]) -> pd.DataFrame:
    """Convert a list of evaluation dicts to a comparison DataFrame."""
    return pd.DataFrame(results).set_index("model")


def confusion_matrix_with_labels(y_true, y_pred, class_names: List[str] = None) -> pd.DataFrame:
    """Build labeled confusion matrix as DataFrame."""
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    return pd.DataFrame(cm,
                         index=[f"True {n}" for n in class_names],
                         columns=[f"Pred {n}" for n in class_names])


def feature_importance_df(feature_names: List[str], importances: np.ndarray,
                           top_k: int = None) -> pd.DataFrame:
    """Build sorted feature importance DataFrame."""
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "abs_importance": np.abs(importances),
    }).sort_values("abs_importance", ascending=False).drop(columns="abs_importance")

    if top_k:
        df = df.head(top_k)
    return df.reset_index(drop=True)


def cross_validation_summary(scores: np.ndarray, metric_name: str = "score") -> dict:
    """Summarize cross-validation results."""
    return {
        "metric": metric_name,
        "mean": scores.mean(),
        "std": scores.std(),
        "min": scores.min(),
        "max": scores.max(),
        "ci_95_lower": scores.mean() - 1.96 * scores.std() / np.sqrt(len(scores)),
        "ci_95_upper": scores.mean() + 1.96 * scores.std() / np.sqrt(len(scores)),
        "n_folds": len(scores),
    }
