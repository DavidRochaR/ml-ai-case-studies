"""
Shared Visualization Utilities
================================

Reusable plotting functions across all case studies.

Author: David Rocha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_distribution(df: pd.DataFrame, column: str,
                       bins: int = 30, ax=None) -> None:
    """Histogram + KDE for a single variable."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column], bins=bins, kde=True, color="steelblue", ax=ax)
    ax.set_title(f"Distribution: {column}", fontweight="bold")


def plot_correlation_heatmap(df: pd.DataFrame, ax=None) -> None:
    """Correlation matrix heatmap for numeric columns."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f",
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix", fontweight="bold")


def plot_confusion_matrix(y_true, y_pred, class_names: List[str] = None,
                           normalize: bool = False, ax=None) -> None:
    """Plot confusion matrix as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""),
                  fontweight="bold")


def plot_roc_curve(y_true, y_prob, ax=None) -> None:
    """ROC curve for binary classification."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")


def plot_feature_importance(importance_df: pd.DataFrame, top_k: int = 15,
                             ax=None) -> None:
    """Horizontal bar chart of feature importance."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    df = importance_df.head(top_k)
    colors = sns.color_palette("viridis", len(df))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color=colors)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Features", fontweight="bold")


def plot_residuals(y_true, y_pred, ax=None) -> None:
    """Residual plot for regression diagnostics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, color="steelblue")
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot", fontweight="bold")


def plot_predicted_vs_actual(y_true, y_pred, ax=None) -> None:
    """Predicted vs Actual scatter plot for regression."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.5, color="teal")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual", fontweight="bold")


def plot_learning_curve(train_scores, val_scores, ax=None) -> None:
    """Training curve for deep learning models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_scores) + 1)
    ax.plot(epochs, train_scores, "b-", label="Train", linewidth=2)
    ax.plot(epochs, val_scores, "r-", label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
