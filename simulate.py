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


# ── COST MODEL ─────────────────────────────────────────────────────────────────
# These numbers approximate real AWS/GCP pricing and SLA penalty structures.

INSTANCE_COST_PER_MINUTE = 0.02    # $0.02 per minute to run one extra EC2 instance
# $0.50 per minute when CPU > 80% (degraded user experience)
SLA_BREACH_PENALTY = 0.50
CPU_SPIKE_THRESHOLD = 80.0    # CPU % above which users notice slowdowns
# Minutes from "spin up" command to instance being live
INSTANCE_BOOT_MINUTES = 3
# Minutes to keep instance after spike before scaling back down
SCALE_DOWN_COOLDOWN = 15
PROACTIVE_TRIGGER_PROB = 0.35    # ML probability threshold to start early scale-up
# Lower than precision threshold (0.78) because here
# we WANT to act earlier and catch more spikes


class ScalingSimulator:

    def __init__(self, name: str):
        self.name = name

        # Server state
        self.extra_instances = 0    # how many extra instances currently running
        self.boot_countdown = 0    # minutes remaining until booting instance is live
        self.cooldown_timer = 0    # minutes remaining before we can scale back down

        # Cumulative metrics (we add to these every minute)
        self.total_sla_breaches = 0
        self.total_breach_cost = 0.0
        self.total_instance_cost = 0.0
        self.scale_up_count = 0
        self.scale_down_count = 0

        # Log every minute's state for charts later
        self.minute_log = []    # list of dicts, one per minute

    def _try_spin_up(self):
        """
        Initiates booting a new instance if one isn't already booting or running.

        The leading underscore _ is Python convention for "private method":
        intended to be called only from within this class, not from outside.

        We only spin up if:
          extra_instances == 0  → no extra instance already running
          boot_countdown  == 0  → no instance currently booting
        This prevents stacking up multiple boot requests for one spike.
        """
        if self.extra_instances == 0 and self.boot_countdown == 0:
            self.boot_countdown = INSTANCE_BOOT_MINUTES
            self.scale_up_count += 1

    def _tick(self, actual_cpu: float) -> tuple:
        """
        Advances the simulation by one minute.
        Called every minute AFTER the subclass makes its decision.

        Returns:
            effective_cpu (float): CPU after load distribution across instances
            breach (int):          1 if this minute was an SLA breach, 0 otherwise
        """
        # ── Progress the boot countdown ──────────────────────────────────────
        if self.boot_countdown > 0:
            self.boot_countdown -= 1
            if self.boot_countdown == 0:
                # Instance is now live! It joins the pool.
                self.extra_instances += 1

        # ── Progress the scale-down cooldown ─────────────────────────────────
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        # ── Compute effective CPU after load distribution ─────────────────────
        # SIMPLIFICATION: we model load as perfectly distributed.
        # 1 instance handles 100% of load.
        # 2 instances (1 extra) each handle 50%: effective = actual / 2
        # 3 instances (2 extra) each handle 33%: effective = actual / 3
        # In reality, load balancing is imperfect — this is a simulation.
        if self.extra_instances > 0:
            effective_cpu = actual_cpu / (1.0 + self.extra_instances)
        else:
            effective_cpu = actual_cpu

        # ── Check for SLA breach ──────────────────────────────────────────────
        breach = 1 if effective_cpu > CPU_SPIKE_THRESHOLD else 0
        self.total_sla_breaches += breach
        self.total_breach_cost += breach * SLA_BREACH_PENALTY

        # ── Accumulate instance running cost ─────────────────────────────────
        # Cost ticks every minute for every extra instance running
        self.total_instance_cost += self.extra_instances * INSTANCE_COST_PER_MINUTE

        # ── Scale down when safe ──────────────────────────────────────────────
        # We scale down only when:
        #   - An extra instance exists to remove
        #   - Effective CPU is comfortably below threshold (< 52%)
        #   - The cooldown has passed (prevents thrashing)
        if (self.extra_instances > 0
                and effective_cpu < CPU_SPIKE_THRESHOLD * 0.65
                and self.cooldown_timer == 0):
            self.extra_instances -= 1
            self.scale_down_count += 1
            self.cooldown_timer = SCALE_DOWN_COOLDOWN

        # ── Log this minute ───────────────────────────────────────────────────
        self.minute_log.append({
            'actual_cpu': round(actual_cpu, 2),
            'effective_cpu': round(effective_cpu, 2),
            'extra_instances': self.extra_instances,
            'sla_breach': breach,
            'booting': int(self.boot_countdown > 0),
        })

        return effective_cpu, breach

    @property
    def total_cost(self) -> float:
        """
        KEY CONCEPT — @property decorator:
            Lets us access total_cost as an ATTRIBUTE (obj.total_cost)
            even though it's computed from two other values.
            Without @property, we'd need to call obj.total_cost() — ugly.
            With @property, it looks like a simple variable but acts like a function.
        """
        return self.total_breach_cost + self.total_instance_cost


class ReactiveScaler(ScalingSimulator):
    """
    REACTIVE: "Act AFTER the problem is already happening."

    Decision rule: if actual CPU right now > 80%, spin up an instance.
    Flaw: the instance takes 3 minutes to boot. During those 3 minutes,
    every minute counts as an SLA breach.
    """

    def __init__(self):
        super().__init__("Reactive Scaling")

    def step(self, actual_cpu: float, **kwargs):
        # **kwargs lets ReactiveScaler.step() accept extra arguments it doesn't need, without crashing — so both classes can be called identically in the same loop.
        # so both step() signatures are compatible in the loop below.
        if actual_cpu > CPU_SPIKE_THRESHOLD:
            self._try_spin_up()
        return self._tick(actual_cpu)


class ProactiveScaler(ScalingSimulator):
    """
    PROACTIVE: "Act BEFORE the problem, using ML prediction."

    Decision rule: if our Random Forest predicts spike probability > 0.35,
    spin up NOW. The 3-minute boot finishes before the spike arrives.

    We also keep the reactive fallback as a safety net: if we missed
    the prediction and CPU is already high, we still respond.
    """

    def __init__(self):
        super().__init__("Proactive Scaling (ML)")

    def step(self, actual_cpu: float, spike_prob: float = 0.0,
             predicted_cpu: float = 0.0):
        # Primary signal: ML spike probability
        if spike_prob > PROACTIVE_TRIGGER_PROB or predicted_cpu > 74:
            self._try_spin_up()
        # Safety fallback: still react if already spiking
        elif actual_cpu > CPU_SPIKE_THRESHOLD:
            self._try_spin_up()
        return self._tick(actual_cpu)


def run_simulation(test_df: pd.DataFrame):
    """
    Iterates through every minute in the test set and runs both simulators.

    KEY CONCEPT — iterrows():
        df.iterrows() yields (index, row) tuples for every row.
        row is a Pandas Series — access values with row['column_name'].
        Slower than vectorised operations but necessary here because
        the simulation has STATE that changes minute by minute
        (we can't parallelise: minute 5's decision depends on minute 4's state).
    """
    print(f"\n⏳ Simulating {len(test_df):,} minutes...")

    reactive = ReactiveScaler()
    proactive = ProactiveScaler()

    for _, row in test_df.iterrows():
        actual_cpu = float(row['cpu_usage'])
        spike_prob = float(row.get('rf_spike_prob', 0))
        predicted_cpu = float(row.get('rf_predicted_cpu', actual_cpu))

        reactive.step(actual_cpu)
        proactive.step(actual_cpu, spike_prob=spike_prob,
                       predicted_cpu=predicted_cpu)

    print("   ✅ Done")
    return reactive, proactive


def print_and_return_results(reactive: ReactiveScaler, proactive: ProactiveScaler) -> dict:
    """Prints side-by-side comparison and returns structured results dict."""
    cost_saved = reactive.total_cost - proactive.total_cost
    breach_reduced = reactive.total_sla_breaches - proactive.total_sla_breaches
    pct_saved = (cost_saved / reactive.total_cost *
                 100) if reactive.total_cost > 0 else 0

    print("\n" + "=" * 62)
    print("SIMULATION RESULTS")
    print("=" * 62)
    print(f"\n{'Metric':<32} {'Reactive':>12} {'Proactive (ML)':>14}")
    print("-" * 60)
    print(f"{'SLA Breach Minutes':<32} {reactive.total_sla_breaches:>12,} {proactive.total_sla_breaches:>14,}")
    print(f"{'SLA Breach Cost ($)':<32} ${reactive.total_breach_cost:>11,.2f} ${proactive.total_breach_cost:>13,.2f}")
    print(f"{'Instance Running Cost ($)':<32} ${reactive.total_instance_cost:>11,.2f} ${proactive.total_instance_cost:>13,.2f}")
    print(f"{'TOTAL COST ($)':<32} ${reactive.total_cost:>11,.2f} ${proactive.total_cost:>13,.2f}")
    print(f"{'Scale-Up Events':<32} {reactive.scale_up_count:>12,} {proactive.scale_up_count:>14,}")
    print("-" * 60)
    print(
        f"\n  💰 Proactive saves: ${cost_saved:,.2f}  ({pct_saved:.1f}% reduction)")
    print(f"  ⚡ SLA breach minutes reduced: {breach_reduced:,}")

    return {
        'reactive': {'strategy': reactive.name,
                     'sla_breach_min': reactive.total_sla_breaches,
                     'sla_breach_cost': round(reactive.total_breach_cost, 2),
                     'instance_cost':   round(reactive.total_instance_cost, 2),
                     'total_cost':      round(reactive.total_cost, 2),
                     'scale_up_events': reactive.scale_up_count},
        'proactive': {'strategy': proactive.name,
                      'sla_breach_min': proactive.total_sla_breaches,
                      'sla_breach_cost': round(proactive.total_breach_cost, 2),
                      'instance_cost':   round(proactive.total_instance_cost, 2),
                      'total_cost':      round(proactive.total_cost, 2),
                      'scale_up_events': proactive.scale_up_count},
        'cost_saved':     round(cost_saved, 2),
        'breach_reduced': breach_reduced
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("PREDICTIVE RESOURCE SCALER — SIMULATION")
    print("=" * 62)

    test_df = pd.read_csv("data/test_with_predictions.csv")
    print(f"📂 Loaded {len(test_df):,} test rows")

    reactive, proactive = run_simulation(test_df)
    results = print_and_return_results(reactive, proactive)

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(reactive.minute_log).to_csv(
        "data/reactive_log.csv",   index=False)
    pd.DataFrame(proactive.minute_log).to_csv(
        "data/proactive_log.csv", index=False)
    with open("data/simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Logs saved. Next step: python visualize.py")
