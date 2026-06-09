#   generate_data.py  —  Creates synthetic server CPU usage data,that we require for our project.
#   we have to create a synthetic dataset as we dont't have real world data for this topic.

#   "Synthetic data" = artificially created data that mimics
#   real-world statistical patterns without being real data.


#   we ware targetting to take 90 days of per-minute CPU readings.[Rows: 90 × 24 × 60 = 129,600 (one row per minute)]

import numpy as np
import pandas as pd
import os


np.random.seed(42)


def generate_server_data(days: int = 90) -> pd.DataFrame:
    """
    Generates realistic-looking server CPU data for a given number of days.

    Parameters:
        days (int): How many days of data to generate. Default = 90.

    Returns:
        pd.DataFrame: A table where each row = 1 minute of server activity.

    HOW THE SIMULATION WORKS:
        CPU = base(45%) + daily_wave + weekly_wave + noise + spikes
        Each component adds a layer of realism.
    """

    total_minutes = days * 24 * 60

    timestamps = pd.date_range(
        start='2024-01-01',
        periods=total_minutes,
        freq='min'            # 'min' = 1 minute [gap between timestamps]
    )

    hour_of_day = timestamps.hour         # array of 129,600 values, each 0–23
    day_of_week = timestamps.dayofweek    # array of 129,600 values, each 0–6

    # Why a sine wave?(same type of explanatoion for cosine wave)
    #   A sine wave (np.sin) oscillates smoothly between -1 and +1.
    #   Real CPU usage rises and falls smoothly throughout the day —
    #   that's also a wave. Sine is the natural mathematical model for it.

    #   np.sin(2π × hour / 24) completes exactly ONE full wave per day.

    #   for our purpose not necessary, but we want the PEAK at ~2 PM
    #   Shifting by -6: (hour - 6) / 24 × 2π moves the peak to 2 PM.(as old peak if not shifted was at 6am)
    daily_wave = np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # To Build the weekly pattern
    # Weekday traffic > weekend traffic for most business applications.
    # We use a cosine wave across 7 days
    weekly_wave = np.cos(2 * np.pi * day_of_week / 7) * 0.15

    # These values — 45% base, ±25% daily swing, ±10% weekly swing — are reasonable approximations for a general web application. not in real world .
    base_cpu = 45 + (daily_wave * 25) + (weekly_wave * 10)

    #   to add random noise:->
    #   np.random.normal(mean, std_dev, size) generates random numbers.
    noise = np.random.normal(0, 5, total_minutes)

    # To inject sudden traffic spikes
    #   These can't be captured by a smooth wave. We inject them manually.
    # np.zeros creates an array of all 0.0 values (no spike at any minute yet)
    spikes = np.zeros(total_minutes)

    spike_starts = np.random.choice(
        total_minutes,
        size=int(total_minutes * 0.02),
        replace=False   # False = no index picked twice (no overlapping spikes)
    )

    for idx in spike_starts:
        # Each spike lasts between 5 and 15 minutes(picking random integer between 5 and 15)
        duration = np.random.randint(5, 15)

        # Each spike adds between +20% and +40% CPU usage (picking random float between 20 and 40)
        magnitude = np.random.uniform(20, 40)

        # min(...) prevents the spike from going past the end of our array
        end = min(idx + duration, total_minutes)

        # Add the spike magnitude to every minute in the spike window
        spikes[idx:end] += magnitude

    # Now we have to combine everything to get the final CPU usage for each minute :->
    cpu_usage = base_cpu + noise + spikes

    # CPU usage can't be negative or above 100%, so we clip it to that range.
    cpu_usage = np.clip(cpu_usage, 5, 100)

    # Request count correlates with CPU: more requests = more CPU used.
    # We model it as: requests = cpu × 50 + noise
    request_count = (cpu_usage * 50) + np.random.normal(0, 200, total_minutes)
    request_count = np.clip(request_count, 0, None).astype(int)

 # Memory usage loosely follows CPU but with its own noise
    memory_usage = 30 + (cpu_usage * 0.4) + \
        np.random.normal(0, 3, total_minutes)
    memory_usage = np.clip(memory_usage, 10, 95)

    # Now at last let us build the DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        # .round(2) -> keep 2 decimal places
        'cpu_usage': cpu_usage.round(2),
        'request_count': request_count,
        'memory_usage': memory_usage.round(2)
    })

    return df


if __name__ == "__main__":

    print("🔧 Generating synthetic server data...")

    # Here-> exist_ok=True means: if data/ already exists, don't raise an error.
    os.makedirs("data", exist_ok=True)

    df = generate_server_data(days=90)

    df.to_csv("data/server_logs.csv", index=False)

    print(f"✅ Generated {len(df):,} rows of server data")
    print(f"   Date range : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   CPU min    : {df['cpu_usage'].min():.1f}%")
    print(f"   CPU max    : {df['cpu_usage'].max():.1f}%")
    print(f"   CPU mean   : {df['cpu_usage'].mean():.1f}%")
    print(f"   Spike rows (CPU > 80%) : {(df['cpu_usage'] > 80).sum():,}")
    print(f"\n📁 Saved → data/server_logs.csv")
    print(f"\n✅ Next step: python data_prep.py")
