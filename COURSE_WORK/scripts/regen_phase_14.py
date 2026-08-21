"""Regenerate the Phase 14 Persistence signoff after an upstream Phase 11
regeneration.

The registry enforces ``RERUN_REASON_REQUIRED`` when the same config
fingerprint is detected, but ``materialize_phase_14`` does not forward a
rerun reason. This driver monkey-patches the registry call inside the
materializer so that it issues a canonical ``REPRODUCIBILITY_CHECK`` rerun,
then invokes the materializer as-is.

Run from the COURSE_WORK directory:

    PYTHONPATH=src python3 scripts/regen_phase_14.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_work.baselines import persistence as persistence_mod
from course_work.experiments.registry import ExperimentRegistry
from course_work.utils.artifacts import get_project_root


RERUN_REASON = "REPRODUCIBILITY_CHECK"


def regenerate(project_root: Path):
    original_register = persistence_mod.register_and_start_persistence_run

    def _register_with_rerun(registry: ExperimentRegistry, config, rerun_reason=None):
        return original_register(registry, config, rerun_reason=RERUN_REASON)

    # materializer calls ``register_and_start_persistence_run(...)`` as a
    # module-global binding; swap it temporarily.
    persistence_mod.register_and_start_persistence_run = _register_with_rerun
    try:
        return persistence_mod.materialize_phase_14(project_root)
    finally:
        persistence_mod.register_and_start_persistence_run = original_register


if __name__ == "__main__":
    project_root = get_project_root()
    print(f"Regenerating Phase 14 signoff at {project_root}", flush=True)
    result = regenerate(project_root)
    print(f"  status      : {result['status']}", flush=True)
    print(f"  run_id      : {result['run_id']}", flush=True)
    print(
        "  phase_11 in : "
        f"{result['input_checksums']['artifacts/dataloaders/phase_11_signoff.json']}",
        flush=True,
    )
    print(f"  outputs     : {len(result['output_checksums'])} files", flush=True)
    print(f"  metrics     : {result['validation_metrics']}", flush=True)
