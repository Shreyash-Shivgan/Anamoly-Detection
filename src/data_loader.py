import pandas as pd
import numpy as np
import os


def load_csv_data(filepath):
    """
    Load raw CSV data and convert it to the format required by the model.
    Output columns:
    - timestamp
    - power_consumption
    """

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading {filepath}: {e}")

    # Case 1: Custom dataset with POWER_DEMAND and datetime
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
        if "POWER_DEMAND" in df.columns:
            df = df.rename(columns={"POWER_DEMAND": "power_consumption"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Case 2: Delhi dataset (legacy support)
    elif "DATE" in df.columns and "POWER_DEMAND" in df.columns:
        df = df.rename(columns={"DATE": "timestamp", "POWER_DEMAND": "power_consumption"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Case 3: Already correct format
    elif "timestamp" in df.columns and "power_consumption" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    else:
        # Fallback for dynamic numeric datasets
        if "timestamp" not in df.columns:
            # Look for a datetime-like column
            date_cols = df.select_dtypes(include=['datetime', 'object']).columns
            if len(date_cols) > 0:
                df = df.rename(columns={date_cols[0]: "timestamp"})
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                raise ValueError(
                    "Dataset must contain a timestamp/date column."
                )

    # Sort data
    df = df.sort_values("timestamp")

    # Reset index
    df = df.reset_index(drop=True)

    return df


def save_csv_data(df, filepath):
    """Save dataframe to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def save_numpy_array(array, filepath):
    """Save numpy array"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, array)


def load_numpy_array(filepath):
    """Load numpy array"""
    return np.load(filepath)