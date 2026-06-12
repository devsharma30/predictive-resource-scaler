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
