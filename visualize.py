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


def chart2_actual_vs_predicted(test_df: pd.DataFrame, save_dir: str = "charts"):
    """
    Chart 2: How well do our models predict future CPU?

    We plot 3 lines on the same axes:
      - Actual future CPU (ground truth — the real answer)
      - Linear Regression prediction
      - Random Forest prediction

    A good model = predicted line closely tracks the actual line.
    Where they diverge = prediction error.

    KEY CONCEPT — Showing only a sample:
        The full test set has ~26,000 rows.
        Plotting all of them creates a solid wall of colour — unreadable.
        We take rows 500–2500 (2000 minutes = ~1.4 days) as a representative sample.
        .reset_index(drop=True) resets the index to 0,1,2... so the x-axis
        shows "minutes in this sample" rather than the original row numbers.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/02_actual_vs_predicted.png"

    sample = test_df.iloc[500:2500].reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle('Model Accuracy: Actual vs Predicted Future CPU',
                 fontsize=15, fontweight='bold')

    for i, (col, label, color) in enumerate([
        ('lr_predicted_cpu', 'Linear Regression', C_LR),
        ('rf_predicted_cpu', 'Random Forest',     C_RF)
    ]):
        ax = axes[i]

        # Plot the ground truth (actual future CPU)
        ax.plot(sample.index, sample['future_cpu_5min'],
                color=C_ACTUAL, linewidth=1.2, alpha=0.9, label='Actual future CPU')

        # Plot model predictions
        if col in sample.columns:
            ax.plot(sample.index, sample[col],
                    color=color, linewidth=1.0, alpha=0.75,
                    linestyle='--', label=f'{label} prediction')

        ax.axhline(y=80, color=C_SPIKE, linestyle=':', linewidth=1.2,
                   alpha=0.6, label='80% threshold')

        ax.set_title(f'{label} — closer dashed line = better model')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_ylim(0, 115)
        ax.legend(loc='upper right')

    axes[1].set_xlabel('Time (minutes in sample)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def chart3_reactive_vs_proactive(reactive_log: list, proactive_log: list,
                                 save_dir: str = "charts"):
    """
    Chart 3: The heart of the project — does our ML prediction actually help?

    This chart shows the SAME spike event seen through two lenses:
      - Red panel: reactive scaling — what happens without ML
      - Green panel: proactive scaling — what happens with ML

    KEY CONCEPT — Finding a "good" window to show:
        We don't just show arbitrary minutes.
        We search the log for a stretch of time that CONTAINS spikes,
        then zoom in on it. This makes the difference between strategies
        visually obvious — no spike = no interesting story.

    KEY CONCEPT — pd.DataFrame(list_of_dicts):
        Each element of minute_log is a dict:
        {'actual_cpu': 87.2, 'effective_cpu': 44.1, 'sla_breach': 1, ...}
        pd.DataFrame([dict1, dict2, ...]) creates a table from these.
        Each key becomes a column name.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/03_reactive_vs_proactive.png"

    r_df = pd.DataFrame(reactive_log)
    p_df = pd.DataFrame(proactive_log)

    # Find a 400-minute window that contains at least one spike
    spike_rows = r_df[r_df['actual_cpu'] > 80].index
    if len(spike_rows) > 0:
        # Pick a spike from the first third of the test period
        center = spike_rows[len(spike_rows) // 4]
        start = max(0, center - 200)
        end = min(len(r_df), center + 200)
    else:
        start, end = 0, 400   # fallback if no spikes found

    r = r_df.iloc[start:end].reset_index(drop=True)
    p = p_df.iloc[start:end].reset_index(drop=True)
    x = r.index

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle('Reactive vs Proactive Scaling — Behaviour During a CPU Spike',
                 fontsize=15, fontweight='bold')

    # ── Panel 1: Raw actual CPU (same for both strategies) ───────────────────
    ax = axes[0]
    ax.plot(x, r['actual_cpu'], color=C_ACTUAL,
            linewidth=1.4, label='Actual CPU (raw)')
    ax.axhline(y=80, color=C_SPIKE, linestyle='--', linewidth=1.5,
               alpha=0.7, label='80% SLA threshold')
    ax.fill_between(x, 80, r['actual_cpu'],
                    where=r['actual_cpu'] > 80,
                    color=C_SPIKE, alpha=0.2, label='Above threshold')
    ax.set_title('Raw Server CPU (before any scaling intervention)')
    ax.set_ylabel('CPU (%)')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')

    # ── Panel 2: Reactive — effective CPU after reactive scaling ─────────────
    ax = axes[1]
    ax.plot(x, r['actual_cpu'], color='gray', linewidth=0.7,
            alpha=0.35, linestyle='--', label='Actual CPU (reference)')
    ax.plot(x, r['effective_cpu'], color=C_REACTIVE, linewidth=1.5,
            label='Effective CPU (reactive)')
    # Shade SLA breach minutes red
    ax.fill_between(x, 80, r['effective_cpu'],
                    where=r['effective_cpu'] > 80,
                    color=C_REACTIVE, alpha=0.3, label='SLA breach ❌')
    ax.axhline(y=80, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_title(
        'REACTIVE: Waits for spike to happen → 3-min boot lag → SLA breaches')
    ax.set_ylabel('CPU (%)')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')

    # ── Panel 3: Proactive — effective CPU after ML-driven scaling ───────────
    ax = axes[2]
    ax.plot(x, p['actual_cpu'], color='gray', linewidth=0.7,
            alpha=0.35, linestyle='--', label='Actual CPU (reference)')
    ax.plot(x, p['effective_cpu'], color=C_PROACTIVE, linewidth=1.5,
            label='Effective CPU (proactive ML)')
    ax.fill_between(x, 80, p['effective_cpu'],
                    where=p['effective_cpu'] > 80,
                    color=C_PROACTIVE, alpha=0.25, label='SLA breach (fewer)')
    ax.axhline(y=80, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_title(
        'PROACTIVE (with ML): ML predicts ahead → instance booted early → spike absorbed')
    ax.set_ylabel('CPU (%)')
    ax.set_xlabel('Time (minutes in window)')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def chart5_feature_importance(save_dir: str = "charts"):
    """
    Chart 5: Which features did the Random Forest rely on most?

    KEY CONCEPT — Horizontal bar chart (barh):
        ax.barh(y_positions, x_values)
        'h' = horizontal — bars go left-to-right instead of bottom-to-top.
        Better for labelling many categories (feature names) on the y-axis
        because long text labels don't overlap each other.

    KEY CONCEPT — Why we colour CPU-related features differently:
        We use green for cpu/lag/rolling features and blue for time features.
        This immediately shows the viewer: "temporal CPU history dominates,
        time of day is secondary" — which is the key insight of our project.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/05_feature_importance.png"

    importance_path = "models/feature_importance.csv"
    if not os.path.exists(importance_path):
        print(
            f"   ⚠️  {importance_path} not found — run train_models.py first")
        return

    imp_df = pd.read_csv(importance_path).head(15)   # top 15 features

    # Colour bars: green if the feature is CPU/lag/rolling related, blue otherwise
    colors = [
        C_RF if any(kw in f for kw in ['cpu', 'lag', 'rolling', 'request'])
        else C_ACTUAL
        for f in imp_df['feature']
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    # invert_yaxis() puts the most important feature at the TOP of the chart
    # Without it, the chart would show least important at the top (confusing)
    bars = ax.barh(imp_df['feature'], imp_df['importance'],
                   color=colors, alpha=0.85, edgecolor='white')
    ax.invert_yaxis()

    # Add exact importance value at the end of each bar
    for bar, val in zip(bars, imp_df['importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlabel('Feature Importance (higher = model relied on it more)')
    ax.set_title('Random Forest Feature Importance\n'
                 'What inputs matter most for predicting CPU spikes?')

    # Manual legend explaining the colour coding
    # Patch = a coloured rectangle for the legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_RF,     label='CPU / lag / rolling features'),
        Patch(facecolor=C_ACTUAL, label='Time / other features'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def chart4_cost_comparison(results: dict, save_dir: str = "charts"):
    """
    Chart 4: The business impact — side-by-side cost comparison.

    KEY CONCEPT — Stacked bar chart:
        A stacked bar groups multiple data categories into one bar.
        Each segment = one cost component (breach penalty vs instance cost).
        The total bar height = total cost.
        Comparing two stacked bars immediately shows which strategy costs more.

    KEY CONCEPT — np.arange(n):
        Returns [0, 1, 2, ..., n-1] as a numpy array.
        Used for bar positions on the x-axis.
        We then offset bars left/right by ±width/2 to create grouped bars.

    KEY CONCEPT — ax.text(x, y, string):
        Places a text label at coordinates (x, y) on the chart.
        We use it to put the actual dollar value on top of each bar.
        ha='center' = horizontal alignment centred on x.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/04_cost_comparison.png"

    r = results['reactive']
    p = results['proactive']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Business Impact: Reactive vs Proactive Scaling Costs',
                 fontsize=15, fontweight='bold')

    # ── LEFT: Stacked cost breakdown ─────────────────────────────────────────
    ax = axes[0]

    categories = ['Reactive', 'Proactive (ML)']
    breach_costs = [r['sla_breach_cost'],  p['sla_breach_cost']]
    instance_costs = [r['instance_cost'],    p['instance_cost']]

    x = np.arange(len(categories))   # [0, 1] — positions of the two bars
    width = 0.45

    # Bottom bars (SLA breach penalty)
    bars1 = ax.bar(x, breach_costs, width,
                   label='SLA Breach Penalty ($)',
                   color=C_REACTIVE, alpha=0.85)

    # Top bars (instance running cost) — stacked on top of breach costs
    # 'bottom=breach_costs' tells matplotlib to start these bars at the
    # top of the breach cost bars, not at zero
    bars2 = ax.bar(x, instance_costs, width,
                   bottom=breach_costs,
                   label='Instance Running Cost ($)',
                   color=C_LR, alpha=0.85)

    # Label each segment with its dollar value
    for bar, val in zip(bars1, breach_costs):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f'${val:.0f}',
                    ha='center', va='center',
                    fontweight='bold', color='white', fontsize=10)

    # Label total on top of each stacked bar
    totals = [r['total_cost'], p['total_cost']]
    for i, total in enumerate(totals):
        ax.text(i, total + (max(totals) * 0.02),
                f'Total\n${total:.0f}',
                ha='center', fontweight='bold', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel('Cost ($)')
    ax.set_title('Total Cost Breakdown by Component')
    ax.legend()

    # ── RIGHT: Key metrics comparison (normalised bar chart) ─────────────────
    ax = axes[1]

    metrics = ['SLA Breach\nMinutes', 'Scale-Up\nEvents', 'Total\nCost ($)']
    r_vals = [r['sla_breach_min'], r['scale_up_events'], r['total_cost']]
    p_vals = [p['sla_breach_min'], p['scale_up_events'], p['total_cost']]

    # Normalise to 0–100 scale so all three metrics fit on the same chart
    # max_vals[i] = the larger of the two strategies for metric i
    max_vals = [max(rv, pv) for rv, pv in zip(r_vals, p_vals)]
    r_norm = [rv / mv * 100 if mv > 0 else 0 for rv,
              mv in zip(r_vals, max_vals)]
    p_norm = [pv / mv * 100 if mv > 0 else 0 for pv,
              mv in zip(p_vals, max_vals)]

    x2 = np.arange(len(metrics))
    width = 0.35

    ax.bar(x2 - width/2, r_norm, width, label='Reactive',
           color=C_REACTIVE,  alpha=0.85)
    ax.bar(x2 + width/2, p_norm, width, label='Proactive (ML)',
           color=C_PROACTIVE, alpha=0.85)

    # Place actual (un-normalised) values above each bar
    for i, (rv, pv) in enumerate(zip(r_vals, p_vals)):
        ax.text(i - width/2, r_norm[i] + 1.5,
                f'{rv:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, p_norm[i] + 1.5,
                f'{pv:.0f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Relative score (lower = better)')
    ax.set_title('Key Metrics Comparison (lower bar = better outcome)')
    ax.legend()

    # Add a savings annotation box in the corner
    savings = results['cost_saved']
    colour = 'lightgreen' if savings > 0 else '#ffcccc'
    sign = '+' if savings > 0 else ''
    ax.annotate(f"ML saves: {sign}${savings:.2f}",
                xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=12, fontweight='bold',
                color='green' if savings > 0 else 'red',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=colour, alpha=0.6))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTIVE RESOURCE SCALER — CHARTS")
    print("=" * 60)

    # ── Chart 1: Raw CPU pattern ──────────────────────────────────────────────
    print("\n📊 Chart 1: Raw CPU pattern...")
    df = pd.read_csv("data/processed_logs.csv")
    chart1_raw_cpu_pattern(df)

    # ── Chart 2: Actual vs Predicted ─────────────────────────────────────────
    print("\n📊 Chart 2: Actual vs Predicted CPU...")
    test_df = pd.read_csv("data/test_with_predictions.csv")
    chart2_actual_vs_predicted(test_df)

    # ── Chart 3: Reactive vs Proactive ───────────────────────────────────────
    print("\n📊 Chart 3: Reactive vs Proactive scaling...")
    reactive_log = pd.read_csv("data/reactive_log.csv").to_dict('records')
    proactive_log = pd.read_csv("data/proactive_log.csv").to_dict('records')
    # .to_dict('records') converts a DataFrame into a list of dicts —
    # one dict per row. That's the format our chart function expects.
    chart3_reactive_vs_proactive(reactive_log, proactive_log)

    # ── Chart 4: Cost comparison ──────────────────────────────────────────────
    print("\n📊 Chart 4: Cost comparison...")
    with open("data/simulation_results.json") as f:
        sim_results = json.load(f)
    chart4_cost_comparison(sim_results)

    # ── Chart 5: Feature importance ───────────────────────────────────────────
    print("\n📊 Chart 5: Feature importance...")
    chart5_feature_importance()

    print("\n✅ All 5 charts saved to charts/")
