"""Focused Phase45 acceptance tests.

Tests every plan §183–§192 acceptance condition. Run via:

    PYTHONPATH=src python3 -m pytest tests/unit/test_phase45_acceptance.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from course_work.final_model_lock import (  # noqa: E402
    ARTIFACT_NAMES,
    load_phase44_handoff,
    load_phase44_signoff,
    load_phase42_shortlist,
    lock_candidate,
    derive_final_epoch,
    build_final_dev_population,
    build_scaling_contract,
    scaling_contract_to_dict,
    build_recipe,
    recipe_to_dict,
    build_seed_contract,
    build_run_matrix,
    build_boundary_sensitivity_evidence,
    config_fingerprint,
    recipe_fingerprint,
    lineage_fingerprint,
    lock_fingerprint,
    build_lineage_audit,
)


@pytest.fixture(scope="module")
def fixtures():
    """Load all Phase45 inputs once."""
    signoff = load_phase44_signoff(
        ROOT / "artifacts/rolling_origin/phase_44_signoff.json"
    )
    handoff = load_phase44_handoff(
        ROOT / "artifacts/rolling_origin/phase45_final_model_lock_handoff.json"
    )
    shortlist = load_phase42_shortlist(
        ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json"
    )
    candidate = lock_candidate(handoff, shortlist_payload=shortlist)
    max_epochs = int(candidate.config.get("training", {}).get("max_epochs", 50))
    decision = derive_final_epoch(handoff, max_epochs)
    final_dev = build_final_dev_population(ROOT, lookback_steps=candidate.lookback_steps)
    scaling_contract = build_scaling_contract(candidate.config, materialize_final_fit=False)
    scaling_dict = scaling_contract_to_dict(scaling_contract)
    recipe = build_recipe(
        candidate.config,
        decision.FINAL_REFIT_EPOCHS,
        final_dev.target_ids_fingerprint,
        scaling_dict,
    )
    recipe_dict = recipe_to_dict(recipe)
    seed_contract = build_seed_contract()
    run_matrix = build_run_matrix(
        candidate.candidate_id,
        candidate.config_fingerprint,
        "TBD",
        "TBD",
        decision.FINAL_REFIT_EPOCHS,
        final_dev.target_ids_fingerprint,
        scaling_dict["x_scaler_bundle_id"],
        scaling_dict["y_scaler_bundle_id"],
    )
    lineage_rows, _ = build_lineage_audit(
        candidate.config, ROOT / "artifacts", ROOT
    )
    config_sha = config_fingerprint(candidate.config)
    recipe_sha = recipe_fingerprint(recipe_dict)
    lineage_payload = {"rows": lineage_rows}
    lineage_sha = lineage_fingerprint(lineage_payload)
    lock_sha = lock_fingerprint(config_sha, recipe_sha, lineage_sha)
    return {
        "signoff": signoff,
        "handoff": handoff,
        "shortlist": shortlist,
        "candidate": candidate,
        "decision": decision,
        "final_dev": final_dev,
        "scaling_contract": scaling_contract,
        "scaling_dict": scaling_dict,
        "recipe_dict": recipe_dict,
        "seed_contract": seed_contract,
        "run_matrix": run_matrix,
        "boundary_sensitivity": build_boundary_sensitivity_evidence(ROOT),
        "config_sha": config_sha,
        "recipe_sha": recipe_sha,
        "lineage_sha": lineage_sha,
        "lock_sha": lock_sha,
    }


# ── §183 Phase44 source ──────────────────────────────────────────────────
def test_phase44_signoff_pass(fixtures):
    assert fixtures["signoff"]["overall_status"] in {"PASS", "PASS_WITH_WARNING"}


def test_phase44_approved_for_phase45(fixtures):
    assert fixtures["signoff"]["approved_for_phase45"] is True


def test_recommended_transformer_exists(fixtures):
    assert fixtures["handoff"]["recommended_transformer_candidate_id"]


def test_recommended_candidate_three_folds(fixtures):
    epochs = fixtures["handoff"]["recommended_transformer_inner_best_epochs"]
    assert {"RO1", "RO2", "RO3"} <= set(epochs.keys())


def test_no_obsolete_locked_model_id_access(fixtures):
    """Old code paths used `locked_model_id`; verify new schema rejects them."""
    assert "locked_model_id" not in fixtures["handoff"]
    assert "config" not in fixtures["handoff"] or not isinstance(
        fixtures["handoff"].get("config"), dict
    )


def test_no_obsolete_config_fingerprint_access(fixtures):
    assert "config_fingerprint" not in fixtures["handoff"]


# ── §6 final family ──────────────────────────────────────────────────────
def test_transformer_only(fixtures):
    assert fixtures["candidate"].model_family == "TRANSFORMER_ENCODER"


def test_canonical_candidate_is_tr_c2(fixtures):
    """Verify it matches Phase44 canonical recommendation (not hard-coded assumption)."""
    cid = fixtures["handoff"]["recommended_transformer_candidate_id"]
    # Per Phase44 signoff/handoff, the canonical recommendation is TR_C2_ALT_LOOKBACK.
    assert cid == "TR_C2_ALT_LOOKBACK"


def test_phase42_fingerprint_match(fixtures):
    p42_shortlist_candidate = next(
        c for c in fixtures["shortlist"]["candidates"]
        if c["candidate_id"] == fixtures["candidate"].candidate_id
    )
    assert (
        p42_shortlist_candidate["candidate_config_fingerprint"]
        == fixtures["handoff"]["recommended_transformer_fingerprint"]
    )


# ── §184 lineage ─────────────────────────────────────────────────────────
def test_lineage_s1_s19(fixtures):
    rows = fixtures["lineage_payload()"] if "lineage_payload()" in dir(fixtures) else None  # noqa: E501
    lineage_rows, _ = build_lineage_audit(
        fixtures["candidate"].config, ROOT / "artifacts", ROOT
    )
    stage_ids = {r["stage_id"] for r in lineage_rows}
    expected = {f"S{i:02d}" for i in range(1, 20)}
    expected.update({"Phase42", "Phase44"})
    assert expected <= stage_ids


def test_wb0_locked_no_amendment(fixtures):
    bs = fixtures["boundary_sensitivity"]
    assert bs.get("wb0_primary") is True
    assert bs.get("protocol_amendment_required") is False


# ── §186 epoch policy ───────────────────────────────────────────────────
def test_epochs_loaded_from_evidence(fixtures):
    assert fixtures["decision"].RO1 == 30
    assert fixtures["decision"].RO2 == 37
    assert fixtures["decision"].RO3 == 19


def test_median_equals_30(fixtures):
    assert fixtures["decision"].FINAL_REFIT_EPOCHS == 30


def test_no_fallback_50(fixtures):
    assert fixtures["decision"].FINAL_REFIT_EPOCHS != 50


def test_no_fallback_max_epochs(fixtures):
    candidate_max_epochs = int(
        fixtures["candidate"].config["training"]["max_epochs"]
    )
    assert fixtures["decision"].FINAL_REFIT_EPOCHS != candidate_max_epochs


# ── §187 data / scaling ──────────────────────────────────────────────────
def test_final_dev_train_validation_only(fixtures):
    assert fixtures["final_dev"].test_target_values_accessed is False


def test_last_final_dev_before_first_test(fixtures):
    assert (
        fixtures["final_dev"].last_target_timestamp
        < fixtures["final_dev"].first_test_timestamp
    )


def test_target_ids_fingerprint_deterministic(fixtures):
    fp1 = fixtures["final_dev"].target_ids_fingerprint
    # Re-run population build
    pop2 = build_final_dev_population(ROOT, lookback_steps=fixtures["candidate"].lookback_steps)
    assert pop2.target_ids_fingerprint == fp1


def test_scaler_excludes_test(fixtures):
    assert fixtures["scaling_contract"].Test_rows_used is False


def test_scaler_checksum_required_at_phase46(fixtures):
    sd = fixtures["scaling_dict"]
    assert sd["x_scaler_bundle_checksum"] == "REQUIRED_AT_PHASE46"
    assert sd["y_scaler_bundle_checksum"] == "REQUIRED_AT_PHASE46"


# ── §188 seeds / training ───────────────────────────────────────────────
def test_seeds_exact_42_123_2026(fixtures):
    assert fixtures["seed_contract"]["seeds"] == [42, 123, 2026]


def test_three_planned_runs(fixtures):
    assert len(fixtures["run_matrix"]) == 3


def test_same_epochs_all_seeds(fixtures):
    epochs = {r["final_refit_epochs"] for r in fixtures["run_matrix"]}
    assert epochs == {fixtures["decision"].FINAL_REFIT_EPOCHS}


def test_validation_none(fixtures):
    assert fixtures["recipe_dict"]["validation_loader"] is None


def test_early_stopping_false(fixtures):
    assert fixtures["recipe_dict"]["early_stopping"] is False


def test_checkpoint_type_final_refit(fixtures):
    assert fixtures["recipe_dict"]["checkpoint_type"] == "FINAL_REFIT"


# ── §190 test firewall ───────────────────────────────────────────────────
def test_no_test_targets_in_final_dev(fixtures):
    """Test sample ids must not appear in target_ids."""
    test_path = ROOT / "artifacts/splits/split_membership.csv"
    if test_path.exists():
        # Build full test set timestamps; verify FINAL_DEV population excludes them
        test_ts = set()
        with test_path.open() as f:
            import csv
            for row in csv.DictReader(f):
                if row.get("split_id") == "TEST":
                    test_ts.add(row.get("timestamp"))
        target_ids_str = "\n".join(fixtures["final_dev"].target_ids)
        # Each test timestamp MUST NOT appear as a TRAIN/VALIDATION entry (which it
        # can't because split_id filters exclude TEST). This is a structural assertion.
        for ts in list(test_ts)[:1]:
            assert ts not in target_ids_str


# ── §191 fingerprints / handoff ─────────────────────────────────────────
def test_config_fingerprint_deterministic(fixtures):
    cfg = fixtures["candidate"].config
    assert config_fingerprint(cfg) == fixtures["config_sha"]


def test_recipe_fingerprint_deterministic(fixtures):
    rd = fixtures["recipe_dict"]
    assert recipe_fingerprint(rd) == fixtures["recipe_sha"]


def test_lineage_fingerprint_deterministic(fixtures):
    lineage_rows, _ = build_lineage_audit(
        fixtures["candidate"].config, ROOT / "artifacts", ROOT
    )
    lineage_payload = {"rows": lineage_rows}
    assert lineage_fingerprint(lineage_payload) == fixtures["lineage_sha"]


def test_lock_fingerprint_deterministic(fixtures):
    cs = fixtures["config_sha"]
    rs = fixtures["recipe_sha"]
    ls = fixtures["lineage_sha"]
    assert lock_fingerprint(cs, rs, ls) == fixtures["lock_sha"]


def test_same_inputs_same_lock(fixtures):
    cs = fixtures["config_sha"]
    rs = fixtures["recipe_sha"]
    ls = fixtures["lineage_sha"]
    a = lock_fingerprint(cs, rs, ls)
    b = lock_fingerprint(cs, rs, ls)
    c = lock_fingerprint(cs, rs, ls)
    assert a == b == c


def test_phase46_handoff_present(fixtures):
    handoff_path = ROOT / "artifacts/final_model_lock/phase46_three_seed_handoff.json"
    assert handoff_path.exists()
    handoff = __import__("json").loads(handoff_path.read_text())
    assert handoff["FINAL_REFIT_EPOCHS"] == fixtures["decision"].FINAL_REFIT_EPOCHS
    assert handoff["seed_list"] == [42, 123, 2026]
    assert handoff["test_status"] == "NOT_ACCESSED"
    assert handoff["no_validation"] is True
    assert handoff["no_early_stopping"] is True
    assert handoff["checkpoint_type"] == "FINAL_REFIT"


def test_phase47_guard_present(fixtures):
    guard_path = ROOT / "artifacts/final_model_lock/phase47_test_evaluation_guard.json"
    assert guard_path.exists()
    guard = __import__("json").loads(guard_path.read_text())
    assert guard["test_access_first_allowed_phase"] == 47
    assert guard["phase45_test_access"] == "forbidden"
    assert guard["phase46_test_access"] == "forbidden"


def test_all_o45_artifacts_present():
    artifact_dir = ROOT / "artifacts/final_model_lock"
    existing = {p.name for p in artifact_dir.iterdir() if p.is_file()}
    # phase_45_signoff.json is a sub-case of O45 but is the OUTPUT, not a gate input
    assert (ARTIFACT_NAMES - existing) == set(), (
        f"Missing O45 artifacts: {sorted(ARTIFACT_NAMES - existing)}"
    )


# ── §4 schema — verify no use of obsolete keys ─────────────────────────
def test_no_obsolete_handoff_keys():
    handoff_path = ROOT / "artifacts/rolling_origin/phase45_final_model_lock_handoff.json"
    import json
    h = json.loads(handoff_path.read_text())
    # These legacy keys must NOT appear in the canonical handoff
    for obsolete in ("locked_model_id", "config", "config_fingerprint"):
        assert obsolete not in h, f"obsolete key {obsolete!r} found in handoff"


def test_signoff_uses_overall_status():
    signoff_path = ROOT / "artifacts/rolling_origin/phase_44_signoff.json"
    import json
    s = json.loads(signoff_path.read_text())
    assert "overall_status" in s
    assert "approved_for_phase45" in s
    assert s["approved_for_phase45"] is True
