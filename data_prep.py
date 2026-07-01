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

# =============================================================================
# PARAMETERS
# =============================================================================

PREDICT_AHEAD = 5       # predict whether a spike happens 5 minutes FROM NOW
SPIKE_THRESHOLD = 80    # CPU above 80% = spike (matches SLA breach threshold)


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # KEY CONCEPT — pd.to_datetime():
    #   Converts string timestamps to datetime objects.
    #   Once a datetime, we can extract .hour, .day_of_week, etc.
    #   Without this, 'timestamp' is just a string — useless to the model.

    print(f"  Loaded {len(df):,} rows")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Time features capture WHEN spikes happen.
    CPU spikes cluster at business hours (hour=9-17), weekdays (day_of_week<5).
    The model uses these to learn: "if it's 2 PM on a Tuesday, spikes are likely."
    """
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek   # 0=Monday, 6=Sunday
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY LAG FEATURES (and why NO raw cpu_usage):
    A single snapshot of CPU at minute T can't tell you if CPU is RISING or FALLING.
    Lag features give the model a "rear-view mirror":
      cpu_lag_1  = CPU 1 minute ago
      cpu_lag_5  = CPU 5 minutes ago
      cpu_lag_10 = CPU 10 minutes ago
      cpu_lag_30 = CPU 30 minutes ago

    If the model sees: cpu_lag_30=45%, cpu_lag_10=55%, cpu_lag_5=65%, cpu_lag_1=72%
    → it can detect the ramp-up trend → predict spike in 5 minutes.

    This is the signal we designed the data generator to produce (Phase 1 = ramp-up).

    KEY CONCEPT — .shift(n):
      Moves every value DOWN by n rows.
      cpu_lag_5 at row 100 = cpu_usage at row 95 = "5 minutes ago"
      The CURRENT cpu_usage is intentionally NOT included as a raw feature
      because it causes data leakage (see file header).
    """
    df['cpu_lag_1'] = df['cpu_usage'].shift(1)
    df['cpu_lag_5'] = df['cpu_usage'].shift(5)
    df['cpu_lag_10'] = df['cpu_usage'].shift(10)
    df['cpu_lag_30'] = df['cpu_usage'].shift(30)

    # Request count lags — helpful since request_count sometimes spikes BEFORE CPU
    df['req_lag_1'] = df['request_count'].shift(1)
    df['req_lag_5'] = df['request_count'].shift(5)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY ROLLING FEATURES:
    Lag features are single point-in-time snapshots.
    Rolling features summarise a WINDOW of recent history — more robust to noise.

      Rolling mean  = average CPU over past N minutes (smoothed trend)
      Rolling std   = how variable CPU has been (rising std = instability)
      Rolling max   = highest CPU seen in past N minutes (worst-case recent)

    KEY CONCEPT — .rolling(window).mean():
      At row 100, rolling_mean_15 = average of rows 86 to 100 (past 15 minutes).
      min_periods=1 prevents NaN at start of dataset (uses whatever data exists).
    """
    # Rolling means at different time horizons
    df['cpu_rolling_mean_5'] = df['cpu_usage'].rolling(5, min_periods=1).mean()
    df['cpu_rolling_mean_15'] = df['cpu_usage'].rolling(
        15, min_periods=1).mean()
    df['cpu_rolling_mean_30'] = df['cpu_usage'].rolling(
        30, min_periods=1).mean()

    # Rolling std — captures volatility
    # WHY: Steady CPU at 70% is very different from CPU oscillating 60-80%.
    # std captures that difference; a rising std often precedes a spike.
    df['cpu_rolling_std_5'] = df['cpu_usage'].rolling(
        5, min_periods=1).std().fillna(0)
    df['cpu_rolling_std_15'] = df['cpu_usage'].rolling(
        15, min_periods=1).std().fillna(0)

    # Rolling max — captures if ANY recent minute was dangerously high
    df['cpu_rolling_max_15'] = df['cpu_usage'].rolling(15, min_periods=1).max()

    # Rolling request count trends
    df['req_rolling_mean_5'] = df['request_count'].rolling(
        5, min_periods=1).mean()

    return df


def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY RATE OF CHANGE:
    This is the most direct feature for detecting a ramp-up.
    cpu_delta_5 = how much CPU has increased in the last 5 minutes.
    If cpu_delta_5 = +15, CPU is rising fast → spike likely.
    If cpu_delta_5 = -5, CPU is falling → no spike.

    This feature directly captures the ramp-up pattern we designed into the data.
    """
    df['cpu_delta_5'] = df['cpu_usage'].diff(
        5).fillna(0)    # change over 5 min
    df['cpu_delta_15'] = df['cpu_usage'].diff(
        15).fillna(0)  # change over 15 min
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: The model needs to know WHAT to predict.
    We want to predict: "Will CPU exceed 80% in the next 5 minutes?"

    KEY CONCEPT — .shift(-n):
      NEGATIVE shift moves values UP by n rows.
      cpu_future_5 at row 100 = cpu_usage at row 105 = "5 minutes from now"
      This is the target. We check if that future CPU > SPIKE_THRESHOLD.
      Result is 0 or 1 (binary classification problem).

    WHY BINARY (0/1) INSTEAD OF EXACT CPU VALUE:
      We don't need to predict exact future CPU — we just need to know
      "will it spike?" to decide whether to boot a new instance.
      Binary classification is a cleaner, more actionable problem.
    """
    df['cpu_future_5'] = df['cpu_usage'].shift(-PREDICT_AHEAD)

    # spike_in_5min = 1 if CPU will be above threshold in 5 minutes, else 0
    df['spike_in_5min'] = (df['cpu_future_5'] > SPIKE_THRESHOLD).astype(int)

    return df


def main():
    print("=" * 60)
    print("data_prep.py — Feature Engineering")
    print("=" * 60)

    os.makedirs('data', exist_ok=True)

    df = load_data('data/server_logs.csv')

    print("Engineering features...")
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_rate_of_change(df)
    df = create_target(df)

    # Drop rows with NaN from lag/rolling operations at start and end of dataset.
    # WHY: First 30 rows have incomplete lags (not enough history yet).
    #      Last 5 rows have NaN targets (no future data to look at).
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    print(
        f"  Dropped {rows_before - rows_after:,} rows (NaN from lags/targets)")

    # Final feature set — these are the columns the model trains on.
    # NOTE: cpu_usage is NOT in this list (intentional — see header for why).
    FEATURE_COLUMNS = [
        # Time features
        'hour', 'minute', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
        # Lag features
        'cpu_lag_1', 'cpu_lag_5', 'cpu_lag_10', 'cpu_lag_30',
        'req_lag_1', 'req_lag_5',
        # Rolling features
        'cpu_rolling_mean_5', 'cpu_rolling_mean_15', 'cpu_rolling_mean_30',
        'cpu_rolling_std_5', 'cpu_rolling_std_15', 'cpu_rolling_max_15',
        'req_rolling_mean_5',
        # Rate of change features (key for ramp-up detection)
        'cpu_delta_5', 'cpu_delta_15',
        # Memory usage (independent signal)
        'memory_usage',
    ]

    print(f"\nFeatures used for training ({len(FEATURE_COLUMNS)} total):")
    for f in FEATURE_COLUMNS:
        print(f"  - {f}")

    output_path = 'data/processed_logs.csv'
    df.to_csv(output_path, index=False)

    spike_rate = df['spike_in_5min'].mean()
    print(f"\n✓ Processed {len(df):,} rows")
    print(f"  Spike rate (target=1): {100*spike_rate:.1f}%")
    print(f"  Non-spike rate (target=0): {100*(1-spike_rate):.1f}%")
    print(f"\n✓ Saved → {output_path}")


if __name__ == '__main__':
    main()
