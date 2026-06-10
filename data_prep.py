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


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates rolling (moving) window statistics.

    KEY CONCEPT — Rolling averages:
        A rolling mean looks at the last N values and takes their average.
        This SMOOTHS OUT noise and reveals the true underlying trend.

        RAW CPU readings: [45, 92, 46, 47, 48]
        The 92 is a 1-minute spike (maybe a GC pause).
        Rolling mean (5): (45+92+46+47+48) / 5 = 55.6
        → 55.6 tells a more honest story about the current CPU "level"

    KEY CONCEPT — .rolling(window=n).mean():
        .rolling(window=5) says "create a sliding window of 5 rows".
        .mean() computes the average of each window.
        min_periods=1 means: even if the window has fewer than n values
        (e.g. the first 4 rows before we have 5), still compute the mean.
        Without min_periods=1, those early rows would be NaN.

    KEY CONCEPT — Standard deviation (std):
        Measures how much values are SPREAD OUT around the mean.
        Low std = CPU is stable.
        High std = CPU is bouncing around = might be about to spike.
        This volatility is a useful signal for the model.
    """
    print("\n⚙️  Adding rolling features...")

    df['cpu_rolling_mean_5'] = df['cpu_usage'].rolling(
        window=5,  min_periods=1).mean()
    df['cpu_rolling_mean_15'] = df['cpu_usage'].rolling(
        window=15, min_periods=1).mean()
    df['cpu_rolling_mean_30'] = df['cpu_usage'].rolling(
        window=30, min_periods=1).mean()

    df['cpu_rolling_std_5'] = df['cpu_usage'].rolling(
        window=5,  min_periods=1).std().fillna(0)
    df['cpu_rolling_std_15'] = df['cpu_usage'].rolling(
        window=15, min_periods=1).std().fillna(0)

    df['cpu_rolling_max_15'] = df['cpu_usage'].rolling(
        window=15, min_periods=1).max()

    print("   ✅ Added: rolling mean/std/max across 5, 15, 30 minute windows")
    return df

# What create_target_variable() does — it is NOT predicting anything.
# It is just labelling the existing data.


def create_target_variable(df: pd.DataFrame,
                           threshold: int = 78,
                           predict_ahead: int = 5) -> pd.DataFrame:
    """
    Creates the TARGET variable — the value the model is trained to predict.

    KEY CONCEPT — What is the target?
        In supervised learning, the "target" (also called label or y) is
        the answer we want the model to learn to predict.
        All other columns are "features" (inputs). The target is the output.

        Our target: "Will CPU exceed 78% exactly 5 minutes from now?"
        Answer: 1 (yes, spike coming) or 0 (no spike coming)

    KEY CONCEPT — .shift(-n) for looking FORWARD:
        Normal .shift(n) looks backward (lag).
        .shift(-n) looks FORWARD in time.

        Original:           After .shift(-5):
        Row 0: CPU=40       Row 0: 35    ← value from row 5
        Row 1: CPU=50       Row 1: 72    ← value from row 6
        ...
        Row N-5: CPU=35     Row N-5: NaN ← no future row exists
        Row N-4: CPU=72     Row N-4: NaN
        ...

        So: df['cpu_usage'].shift(-5).iloc[k]
              = df['cpu_usage'].iloc[k+5]
              = "what CPU will be 5 minutes from now"

    WHY THRESHOLD = 78 (not 80)?
        We set the SPIKE DEFINITION at 78% (slightly below the 80% SLA limit).
        This gives the model slightly earlier warning and improves precision.
    """
    print(
        f"\n⚙️  Creating target variable (spike in {predict_ahead} min, threshold={threshold}%)...")

    future_cpu = df['cpu_usage'].shift(-predict_ahead)

    df['spike_in_5min'] = (future_cpu > threshold).astype(int)
    df['future_cpu_5min'] = future_cpu

    spike_pct = df['spike_in_5min'].mean() * 100
    print(f"   ✅ Target created: spike_in_5min (1 = spike coming, 0 = safe)")
    print(
        f"   📊 {spike_pct:.1f}% of minutes have a spike in the next {predict_ahead} min")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows that have NaN values introduced by lag/shift operations.

    WHY ARE THERE NaNs?
        - cpu_lag_30: the first 30 rows have no "30 minutes ago" data
        - spike_in_5min: the last 5 rows have no "5 minutes in the future" data
        → These rows have NaN in at least one column.

    Most ML models crash on NaN inputs. .dropna() removes every row
    that has ANY NaN value in ANY column.

    WHY NOT fill NaNs instead of dropping?
        Filling lag features with fake values (like 0 or the column mean)
        would corrupt the temporal signal. Dropping is safer here
        because we lose only ~35 rows out of 129,600 — negligible.
    """
    print(f"\n🧹 Cleaning: dropping NaN rows...")
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(
        f"   Rows before: {before:,}  →  after: {after:,}  (removed {before - after})")
    return df


def prepare_features_labels(df: pd.DataFrame):
    """
    Separates the DataFrame into:
        X        → feature matrix  (inputs to the model)
        y_class  → classification target  (0 or 1: will there be a spike?)
        y_reg    → regression target      (exact future CPU value)

    COLUMNS EXCLUDED FROM X:
        - timestamp     : a string/datetime — not a useful number for the model
        - future_cpu_5min : this IS what we're predicting → including it as
                            a feature would be "cheating" (data leakage)
        - spike_in_5min   : same — it's our target label, not an input
        - time_of_day     : categorical string ('morning', 'evening') —
                            needs extra encoding we skip for simplicity
    """
    print(f"\n📦 Preparing feature matrix X and label vector y...")

    feature_cols = [
        'hour', 'minute', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
        'cpu_usage',
        'cpu_lag_1', 'cpu_lag_5', 'cpu_lag_10', 'cpu_lag_30',
        'req_lag_1', 'req_lag_5',
        'cpu_rolling_mean_5', 'cpu_rolling_mean_15', 'cpu_rolling_mean_30',
        'cpu_rolling_std_5', 'cpu_rolling_std_15',
        'cpu_rolling_max_15',
        'request_count', 'memory_usage'
    ]

    X = df[feature_cols]
    y_class = df['spike_in_5min']
    y_reg = df['future_cpu_5min']

    # .shape returns (rows, columns) as a tuple
    print(
        f"   X shape: {X.shape}  ({X.shape[1]} features × {X.shape[0]:,} samples)")
    return X, y_class, y_reg, df


def time_based_split(X, y_class, y_reg, test_ratio: float = 0.2):
    """
    Splits data into training and test sets CHRONOLOGICALLY.

        The model only sees past data. Just like in real life.
        If we shuffled the data randomly, the model could see "future" rows
        during training, which would be unrealistic and lead to overfitting.
    """
    print(
        f"\n✂️  Time-based split: {int((1-test_ratio)*100)}% train / {int(test_ratio*100)}% test...")

    n = len(X)
    split_point = int(n * (1 - test_ratio))    # e.g. 129565 × 0.8 = 103,652

    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_class_train = y_class.iloc[:split_point]
    y_class_test = y_class.iloc[split_point:]
    y_reg_train = y_reg.iloc[:split_point]
    y_reg_test = y_reg.iloc[split_point:]

    print(f"   Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    df = load_data()
    df = check_data_quality(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = create_target_variable(df)
    df = clean_data(df)

    X, y_class, y_reg, df = prepare_features_labels(df)
    X_train, X_test, yct, yc_test, yrt, yr_test = time_based_split(
        X, y_class, y_reg)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_logs.csv", index=False)

    print(f"\n✅ Processed data saved → data/processed_logs.csv")
    print(f"\nNext step: python train_models.py")
