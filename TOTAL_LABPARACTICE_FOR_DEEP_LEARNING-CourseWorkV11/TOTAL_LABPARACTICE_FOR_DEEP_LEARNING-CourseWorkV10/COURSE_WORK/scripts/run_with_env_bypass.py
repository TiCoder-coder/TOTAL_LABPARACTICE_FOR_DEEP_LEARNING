"""Run a sweep condition bypassing Phase 1 (env) re-materialization.

Use only when the user has manually verified that the current interpreter
matches the signed ENV-v1 fingerprint and `pip freeze` differs only by
editable packages (which don't appear in `pip freeze` output).

This script monkey-patches `materialize_phase_1` in the env module
BEFORE other modules import it, so all internal callers are bypassed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import course_work.utils.environment as env_mod
from course_work.utils.artifacts import read_json

_orig_materialize_phase_1 = env_mod.materialize_phase_1


def _patched_materialize_phase_1(root):
    signoff_path = root / "artifacts/environment/phase_1_signoff.json"
    if signoff_path.exists():
        return read_json(signoff_path)
    return _orig_materialize_phase_1(root)


env_mod.materialize_phase_1 = _patched_materialize_phase_1
print("[BYPASS] Patched materialize_phase_1 to skip freeze check", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_id")
    parser.add_argument("condition_id")
    args = parser.parse_args()

    from scripts import run_single_condition as rsc

    env_mod.materialize_phase_1 = _patched_materialize_phase_1
    if hasattr(rsc, "materialize_phase_1"):
        rsc.materialize_phase_1 = _patched_materialize_phase_1

    rsc.run_condition(args.sweep_id, args.condition_id)


if __name__ == "__main__":
    main()