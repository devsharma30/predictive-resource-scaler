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
