#   generate_data.py  —  Creates synthetic server CPU usage data,that we require for our project.
#   we have to create a synthetic dataset as we dont't have real world data for this topic.

#   "Synthetic data" = artificially created data that mimics
#   real-world statistical patterns without being real data.


#   we ware targetting to take 90 days of per-minute CPU readings.[Rows: 90 × 24 × 60 = 129,600 (one row per minute)]
#
#   WHY RAMP-UP MATTERS:
#   Real CPU spikes don't appear from nowhere — traffic builds over minutes
#   before a spike. We simulate this with a clear 20-minute ramp-up phase.
#   This gives the ML model a genuine signal to learn from:
#   "CPU has been climbing steadily for 15 minutes → spike incoming."
#

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# =============================================================================
# PARAMETERS
# =============================================================================

DAYS = 90
MINUTES_PER_DAY = 1440
TOTAL_MINUTES = DAYS * MINUTES_PER_DAY   # 129,600 rows

BASE_CPU = 38.0          # idle baseline (%)
NOISE_STD = 3.0          # gaussian noise (smaller = cleaner ramp-up signal)

SPIKE_PROBABILITY = 0.020   # ~2% of minutes START a new spike event

# Spike shape parameters — 3 phases:
RAMP_UP_MINUTES = 20        # Phase 1: CPU climbs over 20 minutes
# WHY 20: gives model 15 minutes of advance warning
#         before CPU crosses 80% (at ~minute 15 of ramp)
SPIKE_SUSTAIN_MINUTES = 10  # Phase 2: CPU stays at peak
RAMP_DOWN_MINUTES = 15      # Phase 3: CPU falls back to normal

SPIKE_BASE_MAGNITUDE = 45   # spike adds this much on top of current base CPU
# e.g. base=40, magnitude=45 → peak CPU = 85% (above 80%)
SPIKE_MAG_VARIANCE = 10     # ±10 variation so spikes are not all identical


def generate_base_cpu(n: int) -> np.ndarray:
    """
    Creates background CPU: daily sine wave + weekly wave + noise.
    No spikes here — those are injected separately.
    """
    t = np.arange(n)

    daily_wave = 18.0 * np.sin(2 * np.pi * t / MINUTES_PER_DAY - 1.8326)
    weekly_wave = 6.0 * np.sin(2 * np.pi * t /
                               (7 * MINUTES_PER_DAY) - np.pi / 2)
    noise = np.random.normal(0, NOISE_STD, n)

    return BASE_CPU + daily_wave + weekly_wave + noise


def inject_spikes_with_rampup(base_cpu: np.ndarray) -> tuple:
    """
    Injects spike events with three phases: ramp-up → sustain → ramp-down.

    WHY THIS DESIGN:
    The ramp-up phase is the key. During ramp-up, cpu_delta_5 (5-minute rate
    of change) will be +3 to +4 per minute. cpu_rolling_mean_5 will be rising.
    cpu_lag_1, cpu_lag_5 will show an increasing sequence.

    The classifier learns: "rising delta + rising lags + hour in [9-17] = spike soon."
    This gives GENUINE 5-minute-ahead predictability.

    No ramp-up (old design) → model can't predict → leakage → bad project.
    With ramp-up → model learns real pattern → honest predictive signal.

    Returns:
      cpu: the modified cpu array with spikes injected
      spike_mask: True at each minute that is part of a spike event
    """
    n = len(base_cpu)
    extra_cpu = np.zeros(n)      # extra CPU load from spike events
    spike_mask = np.zeros(n, dtype=bool)

    i = 0
    event_len = RAMP_UP_MINUTES + SPIKE_SUSTAIN_MINUTES + RAMP_DOWN_MINUTES

    while i < n - event_len - 30:
        if np.random.random() < SPIKE_PROBABILITY:
            magnitude = SPIKE_BASE_MAGNITUDE + \
                np.random.uniform(-SPIKE_MAG_VARIANCE, SPIKE_MAG_VARIANCE)

            # Phase 1: RAMP-UP — linear increase from 5% to full magnitude
            # Starting at 5% (not 0) makes the ramp visible from the very first minute.
            for j, ramp_val in enumerate(np.linspace(magnitude * 0.08, magnitude, RAMP_UP_MINUTES)):
                extra_cpu[i + j] += ramp_val
                spike_mask[i + j] = True

            # Phase 2: SUSTAIN — hold at peak
            for j in range(SPIKE_SUSTAIN_MINUTES):
                extra_cpu[i + RAMP_UP_MINUTES + j] += magnitude
                spike_mask[i + RAMP_UP_MINUTES + j] = True

            # Phase 3: RAMP-DOWN — linear return to 0
            for j, ramp_val in enumerate(np.linspace(magnitude, 0, RAMP_DOWN_MINUTES)):
                extra_cpu[i + RAMP_UP_MINUTES +
                          SPIKE_SUSTAIN_MINUTES + j] += ramp_val
                spike_mask[i + RAMP_UP_MINUTES +
                           SPIKE_SUSTAIN_MINUTES + j] = True

            # Gap between spikes: at least 30 minutes so events don't overlap
            gap = np.random.randint(30, 80)
            i += event_len + gap
        else:
            i += 1

    cpu = base_cpu + extra_cpu
    cpu = np.clip(cpu, 5, 100)
    return cpu, spike_mask


def generate_request_count(cpu: np.ndarray) -> np.ndarray:
    """
    Request count: correlated with CPU but not perfectly.
    Slightly leads CPU (traffic causes load, not the other way around).
    """
    requests = cpu * 13 + np.random.normal(0, 70, len(cpu))
    return np.clip(requests, 10, 2000).astype(int)


def generate_memory_usage(cpu: np.ndarray) -> np.ndarray:
    """
    Memory: slower-moving, dampened version of CPU + independent noise.
    """
    memory = 32 + 0.32 * cpu + np.random.normal(0, 2.5, len(cpu))
    return np.clip(memory, 20, 92)


def main():
    print("=" * 60)
    print("generate_data.py — Creating synthetic server logs")
    print("=" * 60)

    os.makedirs('data', exist_ok=True)

    print(
        f"Generating {TOTAL_MINUTES:,} rows ({DAYS} days × {MINUTES_PER_DAY} min/day)...")

    base_cpu = generate_base_cpu(TOTAL_MINUTES)
    cpu, spike_mask = inject_spikes_with_rampup(base_cpu)
    requests = generate_request_count(cpu)
    memory = generate_memory_usage(cpu)
    timestamps = pd.date_range(
        start='2024-01-01 00:00:00', periods=TOTAL_MINUTES, freq='min')

    df = pd.DataFrame({
        'timestamp':     timestamps,
        'cpu_usage':     np.round(cpu, 2),
        'request_count': requests,
        'memory_usage':  np.round(memory, 2),
        'is_spike_minute': spike_mask.astype(int),
    })

    path = 'data/server_logs.csv'
    df.to_csv(path, index=False)

    spike_mins = spike_mask.sum()
    high_cpu = (cpu > 80).sum()
    print(f"\n✓ Data generated: {len(df):,} rows")
    print(
        f"  Spike event minutes : {spike_mins:,} ({100*spike_mins/TOTAL_MINUTES:.1f}%)")
    print(
        f"  Minutes above 80%   : {high_cpu:,} ({100*high_cpu/TOTAL_MINUTES:.1f}%)")
    print(f"  CPU mean / max      : {cpu.mean():.1f}% / {cpu.max():.1f}%")
    print(f"\n✓ Saved → {path}")


if __name__ == '__main__':
    main()
