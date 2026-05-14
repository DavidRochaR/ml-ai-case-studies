"""Tests for shared utilities."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from preprocessing import explore_dataset, handle_missing, detect_outliers_iqr, encode_categorical
from evaluation import evaluate_regression, evaluate_classification, compare_models


def test_explore_dataset():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    info = explore_dataset(df)
    assert info["shape"] == (3, 2)
    assert info["missing_values"]["a"] == 1


def test_handle_missing():
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0]})
    filled = handle_missing(df, strategy="median")
    assert filled["a"].isnull().sum() == 0
    assert filled["a"].iloc[2] == 2.0  # median of [1, 2, 4]


def test_outliers():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})
    mask = detect_outliers_iqr(df, "a")
    assert mask.iloc[-1] == True  # 100 is an outlier


def test_encode_categorical():
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    encoded = encode_categorical(df, ["color"], method="onehot")
    assert "color_red" in encoded.columns


def test_evaluate_regression():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.0, 4.1, 4.9])
    result = evaluate_regression(y_true, y_pred, name="test")
    assert result["mae"] < 0.2
    assert result["r2"] > 0.9


def test_evaluate_classification():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1])
    result = evaluate_classification(y_true, y_pred, name="test")
    assert 0 < result["accuracy"] <= 1


def test_compare_models():
    results = [
        {"model": "A", "score": 0.8},
        {"model": "B", "score": 0.9},
    ]
    df = compare_models(results)
    assert "B" in df.index


if __name__ == "__main__":
    test_explore_dataset()
    test_handle_missing()
    test_outliers()
    test_encode_categorical()
    test_evaluate_regression()
    test_evaluate_classification()
    test_compare_models()
    print("All tests passed!")
