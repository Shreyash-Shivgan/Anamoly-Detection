import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def handle_missing_values(df):
    """
    Handle missing values
    """
    df = df.ffill().bfill()
    return df


def select_features(df):
    """
    Select numeric features for anomaly detection
    """

    features = []

    if "power_consumption" in df.columns:
        features.append("power_consumption")

    # If more numeric columns exist, add them
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    # Explicitly exclude metadata/categorical columns if they somehow sneaked in
    exclude_cols = ["name", "datetime", "DATE", "location", "timestamp"]
    
    for col in numeric_cols:
        if col not in features and col not in exclude_cols:
            features.append(col)

    if len(features) == 0:
        raise ValueError("No usable numeric features found")

    return df[features]


def normalize_data(data):
    """
    Normalize data
    """

    scaler = MinMaxScaler()

    normalized = scaler.fit_transform(data)

    return normalized, scaler


def create_sliding_windows(data, window_size):
    """
    Create sliding windows
    """

    X = []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])

    return np.array(X)


def split_dataset(X, test_size=0.2):
    """
    Split train and test data
    """

    split_index = int(len(X) * (1 - test_size))

    X_train = X[:split_index]
    X_test = X[split_index:]

    return X_train, X_test