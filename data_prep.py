# data_prep.py  —>  Load raw data and engineer features

#   Now we do "Feature engineering" = creating new input columns that give the model richer signals to learn from.

# features we will CREATE:
#   Time features   → hour, day_of_week, is_weekend
#   Lag features    → cpu_lag_1, cpu_lag_5 (what was CPU N mins ago?)
#   Rolling features→ cpu_rolling_mean_5  (average CPU over last 5 mins)
#   Target variable → spike_in_5min       (will CPU spike in 5 minutes? 0 or 1)

import pandas as pd
import numpy as np
import os


def load_data(filepath: str = "data/server_logs.csv") -> pd.DataFrame:
    """
    Loads the CSV into a DataFrame and parses the timestamp column.

    WHY parse_dates / pd.to_datetime?
        By default, pd.read_csv reads EVERY column as a plain string.
        The timestamp column would arrive as the string "2024-01-01 00:00:00"
        not as an actual date/time object.
        We need it as a real datetime so we can call .dt.hour, .dt.dayofweek, etc.
    """
    print(f"📂 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   ✅ Loaded {len(df):,} rows | columns: {list(df.columns)}")
    return df


def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Always inspect your data before modelling. Real datasets are messy.
    This function prints a quality report so you know what you're working with.

    WHAT WE CHECK:
        Missing values  → NaN (Not a Number) = missing entry in a cell.
                          If columns have NaNs, most ML models will crash or
                          silently produce wrong results.
        Data types      → Are numbers stored as numbers or as strings?
        Basic stats     → Is the CPU range sane (0–100)? Any negatives?
    """
    print("\n🔍 Data Quality Check:")

    missing = df.isnull().sum()
    print(f"   Missing values:\n{missing.to_string()}")

    # .dtypes shows the data type of each column
    print(f"\n   Data types:\n{df.dtypes.to_string()}")

    # .describe() computes count, mean, std, min, quartiles, max for all numeric cols
    print(f"\n   Statistics:\n{df.describe().round(2).to_string()}")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts useful time information from the timestamp column.

    KEY CONCEPT — The .dt accessor:
        When a Pandas column holds datetime objects, the .dt accessor
        lets you extract components from every row at once.

        df['timestamp'].dt.hour       → array of hours (0–23)
        df['timestamp'].dt.dayofweek  → array of weekday numbers (0=Mon)
        df['timestamp'].dt.month      → array of month numbers (1–12)

        This is vectorised: it runs on the whole column at once,
        much faster than looping row by row.

    WHY THESE FEATURES HELP THE MODEL:
        - CPU at 2 PM is typically higher than at 2 AM
          → the model needs to know the hour
        - Weekday vs weekend has different traffic patterns
          → the model needs day_of_week
    """
    print("\n⚙️  Adding time features...")

    df['hour'] = df['timestamp'].dt.hour         # 0 to 23
    df['minute'] = df['timestamp'].dt.minute       # 0 to 59
    df['day_of_week'] = df['timestamp'].dt.dayofweek    # 0=Monday, 6=Sunday
    df['day_of_month'] = df['timestamp'].dt.day          # 1 to 31
    df['month'] = df['timestamp'].dt.month        # 1 to 12

    #   ML models need numbers, not True/False.
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    print("   ✅ Added: hour, minute, day_of_week, day_of_month, month, is_weekend")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates "lag" features: what was the CPU value N minutes ago?

    KEY CONCEPT — Why lag features?
        The model only sees ONE row at a time when making a prediction.
        Without lag features, it can only see the current minute's data.
        With lag features, it also sees the RECENT HISTORY — the trend.

        EXAMPLE without lags:
            Current CPU = 72%.  Is this going up or down?  Can't tell.

        EXAMPLE with lags:
            Current CPU = 72%.  5 mins ago = 55%.  10 mins ago = 40%.
            → CPU is RISING FAST.  Spike is likely coming.

    KEY CONCEPT — .shift(n):
        .shift(n) moves every value DOWN by n rows.

        Original:       After .shift(1):
        Row 0: CPU=30   Row 0: NaN       ← no "1 minute ago" for the first row
        Row 1: CPU=35   Row 1: 30        ← 1 min ago was row 0 = 30
        Row 2: CPU=40   Row 2: 35        ← 1 min ago was row 1 = 35
        Row 3: CPU=45   Row 3: 40

        So df['cpu_lag_1'].iloc[k] = df['cpu_usage'].iloc[k-1]
        "At minute k, cpu_lag_1 tells you what CPU was at minute k-1"
    """
    print("\n⚙️  Adding lag features...")

    df['cpu_lag_1'] = df['cpu_usage'].shift(1)    # 1 minute ago
    df['cpu_lag_5'] = df['cpu_usage'].shift(5)    # 5 minutes ago
    df['cpu_lag_10'] = df['cpu_usage'].shift(10)   # 10 minutes ago
    df['cpu_lag_30'] = df['cpu_usage'].shift(30)   # 30 minutes ago

    # Also lag the request count — high recent requests predict high future CPU
    df['req_lag_1'] = df['request_count'].shift(1)
    df['req_lag_5'] = df['request_count'].shift(5)

    print("   ✅ Added: cpu_lag_1, cpu_lag_5, cpu_lag_10, cpu_lag_30, req_lag_1, req_lag_5")
    return df
