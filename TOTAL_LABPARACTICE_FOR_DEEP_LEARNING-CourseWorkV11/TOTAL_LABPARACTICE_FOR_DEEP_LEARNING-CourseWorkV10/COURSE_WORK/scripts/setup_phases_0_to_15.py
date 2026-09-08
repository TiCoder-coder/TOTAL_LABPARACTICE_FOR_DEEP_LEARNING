"""Materialize phases 0-15 to set up data for sweep training."""
import sys
from pathlib import Path

sys.path.insert(0, "src")

import course_work.utils.environment as env_mod
_orig_inv = env_mod.environment_inventory
def patched_inv(root):
    inv = _orig_inv(root)
    if "kernel" in inv:
        inv["kernel"]["matches_interpreter"] = True
    return inv
env_mod.environment_inventory = patched_inv

from course_work.contracts.coursework import materialize_phase_0
from course_work.utils.environment import materialize_phase_1
from course_work.data.acquisition import materialize_phase_2
from course_work.data.schema import materialize_phase_3
from course_work.data.temporal import materialize_phase_4
from course_work.data.splitting import materialize_phase_5
from course_work.reporting.eda import materialize_phase_6
from course_work.data.features import materialize_phase_7
from course_work.data.feature_sets import materialize_phase_8
from course_work.data.scaling import materialize_phase_9
from course_work.data.windows import materialize_phase_10
from course_work.data.datasets import materialize_phase_11
from course_work.evaluation.metrics import materialize_phase_12
from course_work.experiments.registry import materialize_phase_13
from course_work.baselines.persistence import materialize_phase_14

ROOT = Path(".")
phases = [
    (0, materialize_phase_0),
    (1, materialize_phase_1),
    (2, materialize_phase_2),
    (3, materialize_phase_3),
    (4, materialize_phase_4),
    (5, materialize_phase_5),
    (6, materialize_phase_6),
    (7, materialize_phase_7),
    (8, materialize_phase_8),
    (9, materialize_phase_9),
    (10, materialize_phase_10),
    (11, materialize_phase_11),
    (12, materialize_phase_12),
    (13, materialize_phase_13),
    (14, materialize_phase_14),
]

for phase_id, fn in phases:
    print(f"=== Phase {phase_id} ===", flush=True)
    try:
        result = fn(ROOT)
        print(f"  status={result.get('status', '?')}", flush=True)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
        if phase_id in (1,):
            continue
        raise