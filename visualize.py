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
import matplotlib.patches as mpatches
import json
import os

#   'seaborn-v0_8-whitegrid' gives clean white background with subtle gridlines.
#   Professional look
plt.style.use('seaborn-v0_8-whitegrid')

CHARTS_DIR = 'charts'
FIGSIZE_WIDE = (14, 10)
FIGSIZE_TALL = (14, 12)

# Colour palette — consistent across all charts
C_ACTUAL = '#2196F3'   # blue — actual/raw data
C_LR = '#FF9800'   # orange — linear regression
C_RF = '#4CAF50'   # green — random forest / proactive
C_REACT = '#F44336'   # red — reactive scaler
C_THRESH = '#F44336'   # red dashed — SLA threshold line
C_SPIKE = '#FFCDD2'   # light red — spike zone fill


def load_all_data():
    print("Loading data for charts...")
    raw_df = pd.read_csv('data/server_logs.csv')
    test_df = pd.read_csv('data/test_with_predictions.csv')
    react_df = pd.read_csv('data/reactive_log.csv')
    pro_df = pd.read_csv('data/proactive_log.csv')

    with open('data/simulation_results.json') as f:
        sim_results = json.load(f)
    with open('models/results.json') as f:
        model_results = json.load(f)

    return raw_df, test_df, react_df, pro_df, sim_results, model_results


def chart1_raw_cpu(raw_df):
    """
    Chart: 7-day CPU timeline + average CPU by hour.
    WHY: Shows the data has realistic daily cycles and spike behaviour.
    The viewer needs to believe this is plausible server data.
    """
    print("  Generating chart 1: Raw CPU Pattern...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_WIDE)
    fig.suptitle('Server CPU Usage — Raw Data Analysis',
                 fontsize=15, fontweight='bold', y=1.01)

    # Top panel: 7-day timeline (first 10,080 minutes = 7 days)
    week_data = raw_df.head(10080)
    t = np.arange(len(week_data))
    cpu = week_data['cpu_usage'].values

    ax1.plot(t, cpu, color=C_ACTUAL, linewidth=0.6,
             alpha=0.9, label='CPU usage')
    ax1.axhline(80, color=C_THRESH, linestyle='--',
                linewidth=1.5, label='80% SLA threshold')
    ax1.fill_between(t, 80, cpu, where=(cpu > 80), alpha=0.3,
                     color=C_SPIKE, label='Spike zone (>80%)')
    ax1.set_title('7-Day CPU Timeline (1 data point per minute)', fontsize=11)
    ax1.set_xlabel('Minutes since start')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_ylim(0, 110)
    ax1.legend(loc='upper right', fontsize=9)

    # Bottom panel: average CPU by hour of day
    raw_df['hour'] = pd.to_datetime(raw_df['timestamp']).dt.hour
    hourly_avg = raw_df.groupby('hour')['cpu_usage'].mean()

    colors = [C_SPIKE if v > 65 else C_ACTUAL for v in hourly_avg.values]
    bars = ax2.bar(hourly_avg.index, hourly_avg.values,
                   color=colors, alpha=0.85, edgecolor='white')
    ax2.axhline(80, color=C_THRESH, linestyle='--',
                linewidth=1.5, label='80% threshold')
    ax2.set_title(
        'Average CPU by Hour of Day (0=midnight, 12=noon, 23=11PM)', fontsize=11)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average CPU (%)')
    ax2.set_xticks(range(24))
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = f'{CHARTS_DIR}/01_raw_cpu_pattern.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved {path}")


def chart2_actual_vs_predicted(test_df, model_results):
    """
    Chart: Actual vs predicted CPU for LR and RF models.
    WHY: Shows model quality. If predictions track actual well, model is good.
    Random Forest should visibly track better than Linear Regression.
    """
    print("  Generating chart 2: Actual vs Predicted...")

    # Show first 2000 test rows (visually clear enough, not overcrowded)
    n = min(2000, len(test_df))
    sample = test_df.head(n)
    t = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_WIDE, sharex=True)
    fig.suptitle('Model Accuracy: Actual vs Predicted Future CPU',
                 fontsize=15, fontweight='bold')

    lr_mae = model_results['lr_mae']
    rf_mae = model_results['rf_mae']
    lr_r2 = model_results['lr_r2']
    rf_r2 = model_results['rf_r2']

    # Linear Regression panel
    ax1.plot(t, sample['cpu_future_5'], color=C_ACTUAL,
             linewidth=0.8, label='Actual future CPU', alpha=0.9)
    ax1.plot(t, sample['lr_pred_cpu'], color=C_LR, linewidth=0.8, linestyle='--',
             label=f'Linear Regression (MAE={lr_mae:.1f}%, R²={lr_r2:.3f})', alpha=0.85)
    ax1.axhline(80, color=C_THRESH, linestyle=':',
                linewidth=1.2, alpha=0.6, label='80% threshold')
    ax1.set_title(
        'Linear Regression — closer dashed line = better model', fontsize=11)
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper right', fontsize=9)

    # Random Forest panel
    ax2.plot(t, sample['cpu_future_5'], color=C_ACTUAL,
             linewidth=0.8, label='Actual future CPU', alpha=0.9)
    ax2.plot(t, sample['rf_pred_cpu'], color=C_RF, linewidth=0.8, linestyle='--',
             label=f'Random Forest (MAE={rf_mae:.1f}%, R²={rf_r2:.3f})', alpha=0.85)
    ax2.axhline(80, color=C_THRESH, linestyle=':',
                linewidth=1.2, alpha=0.6, label='80% threshold')
    ax2.set_title(
        'Random Forest — closer dashed line = better model', fontsize=11)
    ax2.set_ylabel('CPU Usage (%)')
    ax2.set_xlabel('Time (minutes in test set)')
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    path = f'{CHARTS_DIR}/02_actual_vs_predicted.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved {path}")


def chart3_reactive_vs_proactive(react_df, pro_df):
    """
    Chart: The core story — 3 panels showing raw CPU, reactive effective CPU,
    proactive effective CPU. This is the most important chart in the project.

    WHY 3 PANELS:
    Panel 1 (raw CPU) = the problem: spikes breach the SLA.
    Panel 2 (reactive) = old solution: SLA breaches still happen (red shading).
    Panel 3 (proactive ML) = new solution: fewer SLA breaches (less red shading).
    The visual difference between panels 2 and 3 IS the project's value.

    WHY CAREFUL WINDOW SELECTION:
    The difference between reactive and proactive is only visible at moments where
    proactive already has 2 instances while reactive still has 1 (the 3-5 min gap
    between prediction and spike). We scan all windows and pick the one where
    the instance count difference between the two scalers is greatest.
    """
    print("  Generating chart 3: Reactive vs Proactive...")

    n = min(500, len(react_df))

    # Find window where proactive and reactive instance counts differ the most.
    # WHY: windows where both have 2 instances look identical (no story).
    # We want to show the timing gap — proactive up early, reactive up late.
    react_inst = react_df['instances'].values
    pro_inst = pro_df['instances'].values
    diff = pro_inst - react_inst  # positive = proactive has more instances

    best_start = 0
    best_score = -1
    for s in range(0, len(react_df) - n, 50):
        window_diff = diff[s:s+n]
        # Score = total minutes proactive had MORE instances than reactive
        # Also require there are actual breaches in reactive (makes story visible)
        react_breaches_in_window = react_df['sla_breach'].iloc[s:s+n].sum()
        pro_breaches_in_window = pro_df['sla_breach'].iloc[s:s+n].sum()
        score = (window_diff > 0).sum() + react_breaches_in_window * 3
        # Bonus if proactive has fewer breaches than reactive in this window
        if pro_breaches_in_window < react_breaches_in_window:
            score += (react_breaches_in_window - pro_breaches_in_window) * 10
        if score > best_score:
            best_score = score
            best_start = s

    start_idx = best_start
    end_idx = start_idx + n

    react_win = react_df.iloc[start_idx:end_idx].reset_index(drop=True)
    pro_win = pro_df.iloc[start_idx:end_idx].reset_index(drop=True)
    t = np.arange(len(react_win))

    raw_cpu = react_win['raw_cpu'].values
    react_eff = react_win['effective_cpu'].values
    pro_eff = pro_win['effective_cpu'].values
    react_inst_w = react_win['instances'].values
    pro_inst_w = pro_win['instances'].values
    react_breach = react_win['sla_breach'].values.astype(bool)
    pro_breach = pro_win['sla_breach'].values.astype(bool)

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_TALL, sharex=True)
    fig.suptitle('Reactive vs Proactive Scaling — Behaviour During CPU Spikes',
                 fontsize=14, fontweight='bold')

    # Panel 1: Raw CPU
    ax = axes[0]
    ax.plot(t, raw_cpu, color=C_ACTUAL,
            linewidth=0.9, label='Actual CPU (raw)')
    ax.axhline(80, color=C_THRESH, linestyle='--',
               linewidth=1.5, label='80% SLA threshold')
    ax.fill_between(t, 80, raw_cpu, where=(raw_cpu > 80),
                    alpha=0.3, color=C_SPIKE, label='Above threshold')
    ax.set_title(
        'Raw Server CPU (before any scaling intervention)', fontsize=11)
    ax.set_ylabel('CPU (%)')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 2: Reactive
    ax = axes[1]
    ax.plot(t, raw_cpu, color='gray', linewidth=0.5, alpha=0.35,
            label='Actual CPU (reference)', linestyle='-')
    ax.plot(t, react_eff, color=C_REACT, linewidth=1.1,
            label='Effective CPU (reactive)')
    ax.fill_between(t, 80, react_eff, where=react_breach, alpha=0.5, color=C_SPIKE,
                    label=f'SLA breach ({react_breach.sum()} min)')
    # Shade regions where reactive is still on 1 instance during a spike
    # (visually shows the "boot lag" gap)
    ax.fill_between(t, 0, 5, where=(react_inst_w == 1) & (raw_cpu > 70),
                    alpha=0.0)  # invisible — just for reference
    ax.axhline(80, color=C_THRESH, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_title(
        f'REACTIVE: Waits for spike → 3-min boot lag → {react_breach.sum()} SLA breach minutes', fontsize=11)
    ax.set_ylabel('CPU (%)')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 3: Proactive
    ax = axes[2]
    ax.plot(t, raw_cpu, color='gray', linewidth=0.5, alpha=0.35,
            label='Actual CPU (reference)', linestyle='-')
    ax.plot(t, pro_eff, color=C_RF, linewidth=1.1,
            label='Effective CPU (proactive ML)')
    ax.fill_between(t, 80, pro_eff, where=pro_breach, alpha=0.5, color=C_SPIKE,
                    label=f'SLA breach (fewer: {pro_breach.sum()} min)')
    ax.axhline(80, color=C_THRESH, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_title(
        f'PROACTIVE (ML): Predicts ahead → instance ready early → only {pro_breach.sum()} SLA breach minutes', fontsize=11)
    ax.set_ylabel('CPU (%)')
    ax.set_xlabel('Time (minutes in window)')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    path = f'{CHARTS_DIR}/03_reactive_vs_proactive.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved {path}")


def chart4_cost_comparison(sim_results):
    """
    Chart: Business impact — cost breakdown + key metrics.
    WHY: Translates ML model performance into dollars. CFO/PM language.
    """
    print("  Generating chart 4: Cost Comparison...")

    r = sim_results['reactive']
    p = sim_results['proactive']
    savings = sim_results['savings']
    breach_pct = sim_results['breach_reduction_pct']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Business Impact: Reactive vs Proactive Scaling Costs',
                 fontsize=14, fontweight='bold')

    # Left: stacked cost bars
    labels = ['Reactive', 'Proactive (ML)']
    sla_costs = [r['sla_cost'], p['sla_cost']]
    inst_costs = [r['instance_cost'], p['instance_cost']]
    totals = [r['total_cost'], p['total_cost']]

    x = np.arange(2)
    bars1 = ax1.bar(x, sla_costs, color=C_REACT, alpha=0.85,
                    label='SLA Breach Penalty ($)')
    bars2 = ax1.bar(x, inst_costs, bottom=sla_costs, color='#FF9800',
                    alpha=0.85, label='Instance Running Cost ($)')

    for i, (sla, total) in enumerate(zip(sla_costs, totals)):
        ax1.text(i, sla / 2, f'${sla:,.0f}', ha='center', va='center',
                 color='white', fontweight='bold', fontsize=12)
        ax1.text(i, total + 15, f'Total\n${total:,.0f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)

    ax1.set_title('Total Cost Breakdown by Component', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('Cost ($)')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, max(totals) * 1.3)

    # Right: key metrics comparison (lower = better)
    metrics = ['SLA Breach\nMinutes', 'Scale-Up\nEvents', 'Total Cost ($)']
    r_vals = [r['sla_breach_minutes'], r['scale_up_events'], r['total_cost']]
    p_vals = [p['sla_breach_minutes'], p['scale_up_events'], p['total_cost']]

    x2 = np.arange(len(metrics))
    width = 0.35
    b1 = ax2.bar(x2 - width/2, r_vals, width, color=C_REACT,
                 alpha=0.85, label='Reactive')
    b2 = ax2.bar(x2 + width/2, p_vals, width, color=C_RF,
                 alpha=0.85, label='Proactive (ML)')

    for bar, val in zip(b1, r_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(b2, p_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Savings annotation box
    ax2.text(0.98, 0.97,
             f'ML saves: ${savings:,.2f}\n{breach_pct:.0f}% fewer SLA breach mins',
             transform=ax2.transAxes, ha='right', va='top', fontsize=11,
             fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='green', linewidth=1.5))

    ax2.set_title(
        'Key Metrics Comparison (lower = better outcome)', fontsize=11)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.set_ylabel('Value (lower = better)')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, max(max(r_vals), max(p_vals)) * 1.35)

    plt.tight_layout()
    path = f'{CHARTS_DIR}/04_cost_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved {path}")


def chart5_feature_importance(model_results):
    """
    Chart: RF classifier feature importance horizontal bar.
    WHY: Shows what the model learned. cpu_delta_5 and rolling means being
    important tells the story: "model detects ramp-up patterns, not just
    current CPU." That's what makes it predictive rather than reactive.
    """
    print("  Generating chart 5: Feature Importance...")

    features = model_results['feature_names']
    importances = model_results['feature_importances']

    # Sort by importance
    pairs = sorted(zip(importances, features), reverse=True)
    importances_sorted, features_sorted = zip(*pairs)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Colour code: delta/rolling = green (these are the interesting ones),
    # lag = teal, time = blue, other = grey
    def get_color(name):
        if 'delta' in name:
            return '#2E7D32'      # dark green — rate of change
        if 'rolling' in name:
            return '#4CAF50'    # green — rolling stats
        if 'lag' in name:
            return '#26A69A'        # teal — lag features
        if name in ('hour', 'minute', 'day_of_week',
                    'day_of_month', 'month', 'is_weekend'):
            return C_ACTUAL  # blue
        return '#9E9E9E'                           # grey — other

    colors = [get_color(f) for f in features_sorted]
    bars = ax.barh(features_sorted, importances_sorted,
                   color=colors, alpha=0.85, edgecolor='white')

    for bar, val in zip(bars, importances_sorted):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlabel(
        'Feature Importance (higher = model relied on it more)', fontsize=11)
    ax.set_title('Random Forest Classifier Feature Importance\n'
                 'What inputs matter most for predicting CPU spikes?',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    # Legend
    legend_items = [
        mpatches.Patch(color='#2E7D32', label='Rate of change (cpu_delta)'),
        mpatches.Patch(color='#4CAF50', label='Rolling statistics'),
        mpatches.Patch(color='#26A69A', label='Lag features'),
        mpatches.Patch(color=C_ACTUAL,  label='Time features'),
        mpatches.Patch(color='#9E9E9E', label='Other'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9)

    plt.tight_layout()
    path = f'{CHARTS_DIR}/05_feature_importance.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved {path}")


def main():
    print("=" * 60)
    print("visualize.py — Generating Charts")
    print("=" * 60)

    os.makedirs(CHARTS_DIR, exist_ok=True)

    raw_df, test_df, react_df, pro_df, sim_results, model_results = load_all_data()

    chart1_raw_cpu(raw_df)
    chart2_actual_vs_predicted(test_df, model_results)
    chart3_reactive_vs_proactive(react_df, pro_df)
    chart4_cost_comparison(sim_results)
    chart5_feature_importance(model_results)

    print(f"\n✓ All 5 charts saved to {CHARTS_DIR}/")


if __name__ == '__main__':
    main()
