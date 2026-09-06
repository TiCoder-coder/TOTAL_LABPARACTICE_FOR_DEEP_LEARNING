"""Phase 45 — Final Model Lock reusable modules.

This package provides the deterministic, no-train machinery required by
``scripts/phase45_final_model_lock.py``.

Submodules:
    inputs:           Schema-validated readers for Phase44 handoff + signoff.
    candidate_lock:   Lock the recommended Transformer + verify family.
    lineage:          Build the S1–S19 + Phase42 + Phase44 lineage audit.
    epoch_policy:     Compute median(RO1, RO2, RO3) and reject forbidden fallbacks.
    final_dev:        Build the FINAL_DEV_REGION-v1 population and fingerprint.
    scaling_contract: Build the FINAL_SCALING-v1 contract (fit-once, no Test).
    recipe:           Build the FINAL_REFIT_MODE-v1 recipe.
    fingerprints:     Produce deterministic config/recipe/lineage/lock hashes.
    artifacts:        Write all O45.* artifacts.
    consistency:      Plan §192 acceptance checklist.
    preflight:        Run pre-lock preflight gates.
"""

from .inputs import load_phase44_handoff, load_phase44_signoff, load_phase42_shortlist
from .candidate_lock import lock_candidate, LockedCandidate
from .lineage import build_lineage_audit, build_candidate_source_audit, build_boundary_sensitivity_evidence, build_baseline_context_evidence
from .epoch_policy import derive_final_epoch, build_epoch_source_audit, build_epoch_policy_contract
from .final_dev import build_final_dev_contract, build_final_dev_population
from .scaling_contract import build_scaling_contract, scaling_contract_to_dict, fit_final_dev_y_scaler
from .recipe import build_recipe, recipe_to_dict, build_seed_contract, build_run_matrix
from .fingerprints import (
    config_fingerprint, recipe_fingerprint, lineage_fingerprint, lock_fingerprint,
)
from .artifacts import (
    ARTIFACT_NAMES, write_all_o45_artifacts, write_phase45_signoff,
    write_phase46_handoff, write_phase47_guard,
)
from .consistency import run_acceptance_checks, ACCEPTANCE_CHECKS
from .preflight import run_preflight

__all__ = [
    "load_phase44_handoff",
    "load_phase44_signoff",
    "load_phase42_shortlist",
    "lock_candidate",
    "LockedCandidate",
    "build_lineage_audit",
    "build_candidate_source_audit",
    "build_boundary_sensitivity_evidence",
    "build_baseline_context_evidence",
    "derive_final_epoch",
    "build_epoch_source_audit",
    "build_epoch_policy_contract",
    "build_final_dev_contract",
    "build_final_dev_population",
    "build_scaling_contract",
    "scaling_contract_to_dict",
    "fit_final_dev_y_scaler",
    "build_recipe",
    "recipe_to_dict",
    "build_seed_contract",
    "build_run_matrix",
    "config_fingerprint",
    "recipe_fingerprint",
    "lineage_fingerprint",
    "lock_fingerprint",
    "ARTIFACT_NAMES",
    "write_all_o45_artifacts",
    "write_phase45_signoff",
    "write_phase46_handoff",
    "write_phase47_guard",
    "run_acceptance_checks",
    "ACCEPTANCE_CHECKS",
    "run_preflight",
]
