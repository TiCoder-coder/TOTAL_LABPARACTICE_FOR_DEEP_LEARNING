"""Phase 50-B focused tests — Train reference + threshold derivation + leakage."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from course_work.analysis.error_regime_analysis import contract, materialize_b, sources, train_reference, thresholds


def test_upstream_phase49_signoff_pass():
    s = sources.load_phase49_signoff()
    assert s["status"] == "PASS"


def test_phase50_handoff_train_only():
    """Phase 49 -> Phase 50 handoff policy is TRAIN_DERIVED_ONLY."""
    from pathlib import Path
    root = sources.project_root()
    h = json.loads((root / "artifacts/residual_analysis/phase50_handoff.json").read_text())
    assert h["phase50_threshold_policy"] == "TRAIN_DERIVED_ONLY"
    assert h["phase50_target_regime_policy"] == "TRAIN_DERIVED_ONLY"


def test_quantile_method_frozen():
    assert contract.QUANTILE_METHOD == "linear"
    assert contract.QUANTILE_LIBRARY == "numpy"
    assert contract.QUANTILE_FUNCTION == "numpy.quantile"
    assert contract.QUANTILE_DTYPE == "float64"


def test_regime_reference_train_size():
    ref = train_reference.build_regime_reference_train()
    assert ref["n"] == 13670


def test_regime_reference_train_excludes_validation_and_test():
    ref = train_reference.build_regime_reference_train()
    ids = ref["target_ids"]
    val_present = any(13814 <= int(t.split("_")[-1]) <= 16773 for t in ids)
    test_present = any(int(t.split("_")[-1]) >= 16774 for t in ids)
    assert not val_present, "Validation IDs leaked into REGIME_REFERENCE_TRAIN-v1"
    assert not test_present, "Test IDs leaked into REGIME_REFERENCE_TRAIN-v1"


def test_regime_reference_train_sha256_deterministic():
    ref1 = train_reference.build_regime_reference_train()
    ref2 = train_reference.build_regime_reference_train()
    assert ref1["target_ids_sha256"] == ref2["target_ids_sha256"]


def test_regime_reference_train_uses_raw_appliances_wh():
    """All Train target_ids must have raw Appliances Wh loaded."""
    ref = train_reference.build_regime_reference_train()
    missing = [t for t in ref["target_ids"] if t not in ref["raw_appliances"]]
    assert len(missing) == 0


def test_thresholds_train_only_no_test_values_used():
    th = json.loads(Path("artifacts/error_by_regime/regime_thresholds_train_only.json").read_text())
    assert th["test_values_used"] is False
    assert th["validation_values_used"] is False
    assert th["created_before_test_error_join"] is True


def test_thresholds_quantile_values():
    th = json.loads(Path("artifacts/error_by_regime/regime_thresholds_train_only.json").read_text())
    assert th["target_level"]["Q25"] < th["target_level"]["Q75"]
    assert th["extreme_high"]["Q90"] > th["target_level"]["Q75"]
    assert th["change_magnitude"]["Q90_abs_delta"] > 0


def test_thresholds_audit_present():
    audit = Path("artifacts/error_by_regime/regime_threshold_audit.csv")
    assert audit.exists()
    rows = audit.read_text().splitlines()
    header = rows[0]
    for col in (
        "threshold_id",
        "source_split",
        "test_used",
        "validation_used",
        "quantile_method",
    ):
        assert col in header
    for r in rows[1:]:
        assert ",TRAIN," in r, f"non-TRAIN threshold found: {r}"
        assert ",False," in r, f"test/validation used in: {r}"


def test_train_reference_audit_present():
    audit = Path("artifacts/error_by_regime/regime_reference_train_audit.csv")
    assert audit.exists()
    rows = audit.read_text().splitlines()
    assert any("n_target_ids_train_reference,13670,13670,PASS" in r for r in rows)
    assert any("validation_excluded,False,False,PASS" in r for r in rows)
    assert any("test_excluded,False,False,PASS" in r for r in rows)


def test_threshold_fingerprint_present():
    fp = json.loads(Path("artifacts/error_by_regime/regime_threshold_fingerprint.json").read_text())
    assert fp["status"] == "PASS"
    assert fp["quantile_method"] == "linear"
    assert len(fp["threshold_file_sha256"]) == 64
    assert len(fp["train_target_ids_sha256"]) == 64


def test_phase47_unchanged():
    """Phase 47 prediction bundles must remain byte-identical post-Phase50-B."""
    import hashlib
    expected = {
        "artifacts/final_test/phase_47_signoff.json": "80614523b6091e21",
        "artifacts/final_test/prediction_checksums.json": "20c1ddec303fbd80",
        "artifacts/final_test/final_test_population_manifest.json": "b17b4130f8981938",
        "artifacts/final_test/predictions/final_test_predictions_persistence.csv": "7115af1c479b8957",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp), f"{p} mutated"


def test_phase48_unchanged():
    import hashlib
    sha = hashlib.sha256(Path("artifacts/prediction_analysis/phase_48_signoff.json").read_bytes()).hexdigest()
    assert sha.startswith("e8c102d582a35dd2")


def test_phase49_unchanged():
    import hashlib
    expected = {
        "artifacts/residual_analysis/phase_49_signoff.json": "9d5fc659717dbc15",
        "artifacts/residual_analysis/phase50_handoff.json": "ebfe2cdfbdb8523f",
        "artifacts/residual_analysis/residual_long_table.csv": "8418a99110bfda70",
        "artifacts/residual_analysis/residual_wide_table.csv": "931ff9109aef236d",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp), f"{p} mutated"


def test_assert_train_only_fires():
    with pytest.raises(AssertionError):
        contract.assert_train_only("Test truth")


def test_assert_no_test_threshold_fires():
    with pytest.raises(AssertionError):
        contract.assert_no_test_threshold("test_residual_quantile")


def test_contract_seeds_frozen():
    assert contract.contract_seed_list() == ["42", "123", "2026"]
    assert contract.contract_n_test() == 2961
    assert contract.contract_test_population_sha256() == (
        "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
    )


def test_deterministic_q25_value():
    """Q25 must equal 50.0 (Train raw Appliances discrete distribution)."""
    th = json.loads(Path("artifacts/error_by_regime/regime_thresholds_train_only.json").read_text())
    assert th["target_level"]["Q25"] == 50.0
    assert th["target_level"]["Q75"] == 100.0


def test_n_classified_delta_pairs_expected():
    """Train has 13670 target_ids; first per segment has no predecessor -> 13669 classified."""
    th = json.loads(Path("artifacts/error_by_regime/regime_thresholds_train_only.json").read_text())
    assert th["change_magnitude"]["valid_delta_pair_count"] == 13669
