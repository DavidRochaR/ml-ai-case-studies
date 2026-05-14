"""
Shared Preprocessing Utilities
================================

Reusable preprocessing functions across all case studies.

Author: David Rocha
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def explore_dataset(df: pd.DataFrame) -> dict:
    """
    Quick exploratory summary of a dataset.

    Returns
    -------
    dict with shape, missing values, dtypes, and basic statistics.
    """
    return {
        "shape": df.shape,
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
    }


def handle_missing(df: pd.DataFrame, strategy: str = "median",
                   columns: List[str] = None) -> pd.DataFrame:
    """
    Impute missing values.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        'median', 'mean', 'mode', 'drop'.
    columns : list, optional
        Specific columns to impute. Defaults to all numeric.
    """
    df = df.copy()

    if strategy == "drop":
        return df.dropna(subset=columns)

    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    for col in columns:
        if df[col].isnull().any():
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str,
                         multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.

    Returns boolean mask where True indicates outlier.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (df[column] < lower) | (df[column] > upper)


def encode_categorical(df: pd.DataFrame, columns: List[str] = None,
                        method: str = "onehot") -> pd.DataFrame:
    """
    Encode categorical variables.

    Parameters
    ----------
    method : str
        'onehot' or 'label'.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include="object").columns.tolist()

    if method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == "label":
        for col in columns:
            df[col] = df[col].astype("category").cat.codes

    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame = None,
                   method: str = "standard") -> tuple:
    """
    Scale features using StandardScaler or MinMaxScaler.

    Fits on training data only to avoid data leakage.

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, None, scaler


def stratified_split(df: pd.DataFrame, target: str,
                      test_size: float = 0.2,
                      val_size: float = 0.0,
                      random_state: int = 42) -> dict:
    """
    Train/val/test split with stratification.

    Returns
    -------
    dict with X_train, X_test, y_train, y_test (and val if specified).
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() < 20 else None
    )

    result = {"X_train": X_train, "X_test": X_test,
              "y_train": y_train, "y_test": y_test}

    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_train if y_train.nunique() < 20 else None
        )
        result.update({"X_train": X_train, "X_val": X_val,
                       "y_train": y_train, "y_val": y_val})

    return result
