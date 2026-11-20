"""Cleaning and feature building for the P_214 predictive-maintenance project."""
import numpy as np
import pandas as pd

SENSORS = [
    "rpm", "load_pct", "ambient_temp_c", "bearing_temp_c", "motor_temp_c",
    "vib_x_mms", "vib_y_mms", "vib_z_mms", "vib_peak_g", "vib_kurtosis",
    "motor_current_a", "power_kw"
]

# 8 important signals that get rolling features
ROLL_SIGNALS = [
    "vib_total", "vib_x_mms", "vib_y_mms", "vib_z_mms",
    "bearing_temp_c", "motor_temp_c", "vib_kurtosis", "motor_current_a"
]

# columns that reveal the answer: never given to the model
LEAKY = [
    "rul_hours", "health_index", "operating_hours",
    "failure_within_48h", "maintenance_class"
]

CLASSES = ["Normal", "Warning", "Critical"]

def load_and_clean(path="predictive_maintenance_dataset.csv"):
    """Read the CSV and fill missing sensor readings machine by machine."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    df[SENSORS] = df.groupby("machine_id")[SENSORS].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    return df

def add_features(df):
    """Add total vibration, temperature gap and rolling features (per machine)."""
    df = df.copy()
    df["vib_total"] = np.sqrt(df.vib_x_mms**2 + df.vib_y_mms**2 + df.vib_z_mms**2)
    df["temp_gap"] = df.bearing_temp_c - df.ambient_temp_c
    g = df.groupby("machine_id")
    for c in ROLL_SIGNALS:
        df[c + "_m6"] = g[c].transform(lambda s: s.rolling(6, min_periods=1).mean())
        df[c + "_m24"] = g[c].transform(lambda s: s.rolling(24, min_periods=1).mean())
        df[c + "_s24"] = g[c].transform(lambda s: s.rolling(24, min_periods=1).std()).fillna(0)
        df[c + "_d12"] = g[c].transform(lambda s: s.diff(12)).fillna(0)
    return df

def feature_columns(df):
    """Model inputs = sensors + engineered columns. Leaky columns are excluded."""
    cols = SENSORS + ["vib_total", "temp_gap"]
    for c in ROLL_SIGNALS:
        cols += [c + "_m6", c + "_m24", c + "_s24", c + "_d12"]
    assert not set(cols) & set(LEAKY)
    return cols

def make_labels(df):
    """Numeric targets for the models."""
    y_cls = df["maintenance_class"].map({c: i for i, c in enumerate(CLASSES)}).values
    y_48 = df["failure_within_48h"].values
    y_rul = df["rul_hours"].values
    return y_cls, y_48, y_rul
