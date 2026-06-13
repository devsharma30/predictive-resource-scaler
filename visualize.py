# visualize.py  —  Generate all 5 charts from the results

# THE 5 CHARTS:
#   1. Raw CPU pattern        → shows what the data looks like
#   2. Actual vs Predicted    → shows how accurate our models are
#   3. Reactive vs Proactive  → the main story: ML prevents spike damage
#   4. Cost comparison        → the business impact in dollar terms
#   5. Feature importance     → which inputs the model relied on most


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os


#   Matplotlib has built-in style sheets that change the look of all charts.
plt.style.use('seaborn-v0_8-whitegrid')
#   rcParams is a dict of default settings for all matplotlib figures.
plt.rcParams['font.family'] = 'DejaVu Sans'   # clean readable font
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'


# Hex colour codes (#RRGGBB). Keeping colours consistent across all charts
C_ACTUAL = '#2196F3'   # blue   — actual/real CPU values
C_LR = '#FF9800'   # orange — Linear Regression predictions
C_RF = '#4CAF50'   # green  — Random Forest predictions
C_REACTIVE = '#F44336'   # red    — reactive scaling (the bad strategy)
C_PROACTIVE = '#4CAF50'   # green  — proactive scaling (the good strategy)
C_SPIKE = '#FF5252'   # bright red — spike threshold / breach zones


def chart1_raw_cpu_pattern(df: pd.DataFrame, save_dir: str = "charts"):
    """
    Chart 1: to understand the raw data before modelling anything.

    Our raw data tells us :->
    - Is the range sensible? (5% to 100% )
    - Are there obvious patterns? (daily cycles )
    - Are there spikes? (yes )
    - Is there noise? (yes )

    TWO SUBPLOTS:
    Top:    7-day CPU timeline (shows the daily wave pattern + spikes)
    Bottom: Average CPU by hour of day (shows which hours are dangerous)

    KEY CONCEPT — fig, axes = plt.subplots(rows, cols):
        Creates a Figure (the entire image) and an array of Axes (subplots).
        subplots(2, 1) → 2 rows, 1 column = 2 charts stacked vertically.
        figsize=(14, 8) → 14 inches wide, 8 inches tall at screen resolution.
        axes[0] = top chart,  axes[1] = bottom chart.

    KEY CONCEPT — ax.fill_between(x, y1, y2, where=condition):
        Fills the area BETWEEN y1 and y2, but only WHERE condition is True.
        We use it to shade the "danger zone" (CPU above 80% threshold)
        in red so spikes are immediately visible.

    KEY CONCEPT — df.groupby('hour')['cpu_usage'].mean():
        groupby('hour')   → splits the DataFrame into 24 groups (one per hour)
        ['cpu_usage']     → selects only the cpu_usage column
        .mean()           → computes the mean for each group
        Result: a Series with index 0-23 and mean CPU for each hour.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/01_raw_cpu_pattern.png"

    # we take only the first 7 days (7 × 24 × 60 = 10,080 rows).
    # if we plotted all 90 days would create 129,600 points — too dense to read.
    week = df.iloc[:60 * 24 * 7].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Server CPU Usage — Raw Data Analysis',
                 fontsize=15, fontweight='bold')

    # ── TOP: 7-day CPU timeline ───────────────────────────────────────────────
    ax = axes[0]

    ax.plot(week.index, week['cpu_usage'],
            color=C_ACTUAL, linewidth=0.8, alpha=0.85, label='CPU usage')

    # Horizontal dashed line at the 80% threshold
    ax.axhline(y=80, color=C_SPIKE, linestyle='--', linewidth=1.5,
               label='80% SLA threshold')

    # Shade everything ABOVE 80% in red — these are the "danger zones"
    ax.fill_between(week.index, 80, week['cpu_usage'],
                    where=week['cpu_usage'] > 80,
                    color=C_SPIKE, alpha=0.25, label='Spike zone (>80%)')

    ax.set_title('7-Day CPU Timeline (1 data point per minute)')
    ax.set_ylabel('CPU Usage (%)')
    ax.set_xlabel('Minutes since start')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')

    # ── BOTTOM: Average CPU by hour ───────────────────────────────────────────
    ax = axes[1]

    hourly_avg = df.groupby('hour')['cpu_usage'].mean()

    # Create a bar for each hour, colour bars red if avg CPU > 60%
    bar_colors = [C_SPIKE if v > 60 else C_ACTUAL for v in hourly_avg.values]
    ax.bar(hourly_avg.index, hourly_avg.values,
           color=bar_colors, alpha=0.75, edgecolor='white', linewidth=0.5)

    ax.axhline(y=80, color=C_SPIKE, linestyle='--', linewidth=1.5,
               alpha=0.7, label='80% threshold')

    ax.set_title('Average CPU by Hour of Day (0=midnight, 12=noon, 23=11PM)')
    ax.set_ylabel('Average CPU (%)')
    ax.set_xlabel('Hour of Day')
    ax.set_xticks(range(0, 24))    # show all 24 hour labels on x-axis
    ax.set_ylim(0, 100)
    ax.legend()

    # tight_layout() automatically adjusts spacing so titles/labels don't overlap
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # dpi=150       → dots per inch, controls image sharpness
    # bbox_inches='tight' → crop whitespace around the figure
    plt.close()
    print(f"   ✅ Saved: {save_path}")
