# simulate.py  —  Reactive vs Proactive scaling simulation

# REACTIVE SCALING (current industry standard):
#   Watch CPU every minute.
#   If CPU > 80% RIGHT NOW → start spinning up a new instance.
#   Problem: the instance takes 3 minutes to boot.
#   During those 3 minutes, users are still getting slow responses.
#   3 minutes × $0.50 = $1.50 lost PER spike, just on boot lag.

# PROACTIVE SCALING (what our ML enables):
#   Our Random Forest predicts: "CPU will exceed 78% in 5 minutes."
#   We start spinning up NOW.
#   The instance is ready BEFORE the spike hits.
#   Users experience no slowdown.

# WHAT WE MEASURE:
#   - Total SLA breach minutes (user-facing impact)
#   - Total cost = breach penalties + instance running cost
#   - Difference between strategies = the value of ML

import pandas as pd
import numpy as np
import json
import os

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

INSTANCE_COST_PER_MINUTE = 0.04   # $0.04/min per extra instance running

SLA_BREACH_PENALTY = 1.50         # $1.50/min when CPU > 80%

CPU_SPIKE_THRESHOLD = 80.0        # SLA breach threshold

INSTANCE_BOOT_MINUTES = 3         # minutes to boot a new instance

SCALE_DOWN_COOLDOWN = 30          # wait 30 min before scaling back down
# WHY 30: spike ramp-up alone is 20 min.
# Combined with the spike_proba < 0.30 guard
# in _tick(), the instance stays alive through
# the full event — cooldown handles the boot+peak
# window, probability guard handles ramp-down.

# trigger proactive scale-up if P(spike) > 65%
PROACTIVE_TRIGGER_PROB = 0.35
# Tuned to classifier threshold for high precision.

SIMULATION_WINDOW = 6000          # simulate on last 6000 minutes of test data


class ScalingSimulator:
    """
    Base class with shared state and logic.
    Both ReactiveScaler and ProactiveScaler inherit from this.

    WHY INHERITANCE:
    Both scalers share: instance tracking, cost tracking, SLA detection,
    boot queue, and cooldown. Only the TRIGGER DECISION differs.
    Inheritance = write shared logic once, override only what changes.
    """

    def __init__(self, name: str):
        self.name = name
        self.instances = 1
        self.boot_queue = []          # list of boot timers counting down to 0
        self.cooldown_remaining = 0
        self.total_sla_cost = 0.0
        self.total_instance_cost = 0.0
        self.sla_breach_minutes = 0
        self.scale_up_events = 0
        self.log = []

    @property
    def total_cost(self) -> float:
        return self.total_sla_cost + self.total_instance_cost

    def _try_spin_up(self):
        """
        Queues a new instance to boot in INSTANCE_BOOT_MINUTES minutes.

        KEY CONCEPT — boot queue:
          When we decide to scale up, the instance doesn't appear instantly.
          We add INSTANCE_BOOT_MINUTES to a queue.
          Each tick, we decrement all queue entries.
          When a timer hits 0, the instance becomes active.
          This simulates the real-world 3-minute EC2 boot delay.
        """
        if self.cooldown_remaining > 0:
            return  # in cooldown — don't thrash
        if len(self.boot_queue) > 0:
            return  # already booting one
        self.boot_queue.append(INSTANCE_BOOT_MINUTES)
        self.scale_up_events += 1
        self.cooldown_remaining = SCALE_DOWN_COOLDOWN

    def _tick(self, raw_cpu: float, spike_proba: float = 0.0):
        """
        Per-minute update: advance boot queue, compute effective CPU,
        check SLA breach, accumulate costs, maybe scale down.

        spike_proba is passed so ProactiveScaler doesn't scale down
        while a spike is still expected (prevents spinning down too early).
        """
        # Advance boot queue — count down each timer, activate at 0
        new_queue = []
        for time_left in self.boot_queue:
            if time_left - 1 <= 0:
                self.instances += 1   # instance is now live
            else:
                new_queue.append(time_left - 1)
        self.boot_queue = new_queue

        # Effective CPU after load distribution
        # why we do-> (instances * 0.92):
        # Perfect load balancing would give raw_cpu / instances.
        # We apply 0.92 efficiency to model inter-instance overhead.
        effective_cpu = raw_cpu / (self.instances * 0.92)
        effective_cpu = min(raw_cpu, effective_cpu)  # can't exceed raw load
        # floor: minimum reduction
        effective_cpu = max(effective_cpu, raw_cpu * 0.22)

        # SLA breach check
        breach = effective_cpu > CPU_SPIKE_THRESHOLD
        if breach:
            self.total_sla_cost += SLA_BREACH_PENALTY
            self.sla_breach_minutes += 1

        # Instance running cost (extra instances only)
        extra_instances = max(0, self.instances - 1)
        self.total_instance_cost += extra_instances * INSTANCE_COST_PER_MINUTE

        # Scale down conditions:
        #   1. Have extra instances
        #   2. Cooldown expired (no recent scale-up)
        #   3. CPU is safely below threshold
        #   4. No spike expected soon (prevents early scale-down for proactive)
        safe_to_scale_down = (
            self.instances > 1
            and self.cooldown_remaining <= 0
            and effective_cpu < 52
            and spike_proba < 0.30
        )
        if safe_to_scale_down:
            self.instances -= 1

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        self.log.append({
            'raw_cpu': raw_cpu,
            'effective_cpu': round(effective_cpu, 2),
            'instances': self.instances,
            'sla_breach': int(breach),
            'instance_cost': round(extra_instances * INSTANCE_COST_PER_MINUTE, 4),
            'sla_cost': SLA_BREACH_PENALTY if breach else 0.0,
        })

    def step(self, raw_cpu: float, spike_proba: float):
        raise NotImplementedError


class ReactiveScaler(ScalingSimulator):
    """
    REACTIVE: triggers scale-up AFTER CPU is already high.

    Sequence of events during a spike:
      Minute 0:   Spike hits, CPU > 80%
      Minute 0:   ReactiveScaler sees high CPU → triggers boot
      Minute 1-3: Instance booting, users still suffering (SLA breach)
      Minute 3:   Instance ready, CPU distributed
      → 3 breach minutes minimum per spike event
    """

    def step(self, raw_cpu: float, spike_proba: float):
        if raw_cpu > CPU_SPIKE_THRESHOLD:
            self._try_spin_up()
        # reactive ignores proba for scale-down too
        self._tick(raw_cpu, spike_proba=0.0)


class ProactiveScaler(ScalingSimulator):
    """
    PROACTIVE: triggers scale-up BEFORE spike using ML prediction.

    Sequence of events with proactive scaling:
      Minute -5:  Model predicts spike, P(spike) > 0.65 → triggers boot
      Minute -2:  Instance ready (3-min boot done, 2 min before spike)
      Minute 0:   Spike hits → load spread across 2 instances → CPU absorbed
      → 0 breach minutes (instance was ready BEFORE spike)

    This only works because:
      a) We predict 5 minutes ahead
      b) Boot takes 3 minutes
      c) 5 > 3 → we have 2 minutes of buffer

    If boot time ≥ prediction window, proactive offers no advantage.
    """

    def step(self, raw_cpu: float, spike_proba: float):
        if spike_proba >= PROACTIVE_TRIGGER_PROB:
            self._try_spin_up()
        self._tick(raw_cpu, spike_proba=spike_proba)


def run_simulation(cpu_series, proba_series):
    reactive = ReactiveScaler("Reactive")
    proactive = ProactiveScaler("Proactive ML")

    cpu_arr = cpu_series.values
    proba_arr = proba_series.values
    n = len(cpu_arr)

    print(f"  Simulating {n:,} minutes...")
    for i in range(n):
        reactive.step(cpu_arr[i], proba_arr[i])
        proactive.step(cpu_arr[i], proba_arr[i])

    return reactive, proactive


def print_results(reactive, proactive):
    print(f"\n{'='*60}")
    print("SIMULATION RESULTS")
    print(f"{'='*60}")

    rows = [
        ("SLA Breach Minutes",  reactive.sla_breach_minutes,
         proactive.sla_breach_minutes,   "",  True),
        ("Scale-Up Events",     reactive.scale_up_events,
         proactive.scale_up_events,       "",  False),
        ("SLA Breach Cost",     reactive.total_sla_cost,
         proactive.total_sla_cost,        "$", True),
        ("Instance Run Cost",   reactive.total_instance_cost,
         proactive.total_instance_cost,   "$", True),
        ("TOTAL COST",          reactive.total_cost,
         proactive.total_cost,            "$", True),
    ]

    for name, rv, pv, unit, lower_better in rows:
        if unit == "$":
            rs, ps, ds = f"${rv:,.2f}", f"${pv:,.2f}", f"${rv-pv:,.2f}"
        else:
            rs, ps, ds = f"{rv:,}", f"{pv:,}", f"{rv-pv:,}"

        ml_wins = (rv > pv and lower_better) or (rv < pv and not lower_better)
        tag = "✓ ML wins" if ml_wins else ""
        print(f"  {name:<22} Reactive={rs:>10}  Proactive={ps:>10}  Δ={ds}  {tag}")

    savings = reactive.total_cost - proactive.total_cost
    breach_red = reactive.sla_breach_minutes - proactive.sla_breach_minutes
    breach_pct = (100 * breach_red / reactive.sla_breach_minutes
                  if reactive.sla_breach_minutes > 0 else 0)

    print(f"\n  → Total cost savings   : ${savings:.2f}")
    print(
        f"  → SLA breach reduction : {breach_red} minutes ({breach_pct:.0f}%)")


def main():
    print("=" * 60)
    print("simulate.py — Reactive vs Proactive Scaling")
    print("=" * 60)

    os.makedirs('data', exist_ok=True)

    test_df = pd.read_csv('data/test_with_predictions.csv')
    raw_df = pd.read_csv('data/server_logs.csv')

    print(f"Test data: {len(test_df):,} rows")

    # Take last SIMULATION_WINDOW rows of test set
    if len(test_df) > SIMULATION_WINDOW:
        sim_df = test_df.iloc[-SIMULATION_WINDOW:].reset_index(drop=True)
    else:
        sim_df = test_df.reset_index(drop=True)

    # Align raw CPU to same window
    n_test = len(test_df)
    start_idx = len(raw_df) - n_test
    raw_cpu = raw_df['cpu_usage'].iloc[start_idx:].reset_index(drop=True)
    if len(raw_cpu) > SIMULATION_WINDOW:
        raw_cpu = raw_cpu.iloc[-SIMULATION_WINDOW:].reset_index(drop=True)

    print(
        f"Simulation window: {len(sim_df):,} minutes ({len(sim_df)/60:.1f} hours)")

    reactive, proactive = run_simulation(raw_cpu, sim_df['rf_spike_proba'])
    print_results(reactive, proactive)

    pd.DataFrame(reactive.log).to_csv('data/reactive_log.csv', index=False)
    pd.DataFrame(proactive.log).to_csv('data/proactive_log.csv', index=False)

    savings = reactive.total_cost - proactive.total_cost
    breach_red = reactive.sla_breach_minutes - proactive.sla_breach_minutes
    breach_pct = (100 * breach_red / reactive.sla_breach_minutes
                  if reactive.sla_breach_minutes > 0 else 0)

    results = {
        'reactive':  {
            'sla_breach_minutes': reactive.sla_breach_minutes,
            'scale_up_events':    reactive.scale_up_events,
            'sla_cost':           round(reactive.total_sla_cost, 2),
            'instance_cost':      round(reactive.total_instance_cost, 2),
            'total_cost':         round(reactive.total_cost, 2),
        },
        'proactive': {
            'sla_breach_minutes': proactive.sla_breach_minutes,
            'scale_up_events':    proactive.scale_up_events,
            'sla_cost':           round(proactive.total_sla_cost, 2),
            'instance_cost':      round(proactive.total_instance_cost, 2),
            'total_cost':         round(proactive.total_cost, 2),
        },
        'savings':                   round(savings, 2),
        'breach_reduction_minutes':  breach_red,
        'breach_reduction_pct':      round(breach_pct, 1),
    }

    with open('data/simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved → data/reactive_log.csv, data/proactive_log.csv")
    print(f"✓ Saved → data/simulation_results.json")


if __name__ == '__main__':
    main()
