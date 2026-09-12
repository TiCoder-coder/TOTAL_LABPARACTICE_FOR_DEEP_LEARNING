from __future__ import annotations

from pathlib import Path
import csv
import json
from types import SimpleNamespace

import pytest

from course_work.model_improvement_v2 import step16_final_refit as step16


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def preflight():
    return step16.run_preflight(ROOT)


def test_epoch_policy_is_common_development_locked_epoch(preflight):
    assert preflight["epoch_policy"]["fold_best_epochs"] == {"RO1": 16, "RO2": 13, "RO3": 23}
    assert preflight["epoch_policy"]["policy"] == "MEDIAN_RO_INNER_BEST_EPOCHS-v1"
    assert preflight["epochs_by_seed"] == {"42": 16, "123": 16, "2026": 16}


def test_full_pretest_population_and_firewall(preflight):
    population = preflight["data_population"]
    assert population["included_splits"] == ["TRAIN", "VALIDATION"]
    assert population["excluded_splits"] == ["TEST"]
    assert population["target_count"] == 16630
    assert population["first_target_id"] == "TGT_00000144"
    assert population["last_target_id"] == "TGT_00016773"
    assert preflight["test_rows_read"] == 0
    assert preflight["test_target_ids_seen"] == 0
    assert preflight["test_status"] == "NOT_ACCESSED"


def test_final_scaling_policy_and_locked_hashes(preflight):
    scaler = preflight["scaler_policy"]
    assert scaler["fit_region"] == "V2_FINAL_DEV_REGION-v1"
    assert scaler["fit_target_count"] == 16630
    assert scaler["fit_once_shared_across_seeds"] is True
    assert scaler["x_scaled_feature_count"] == 28
    assert scaler["x_passthrough_feature_count"] == 5
    assert scaler["passthrough_features"] == ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend"]
    assert all(len(scaler[name]) == 64 for name in ("x_scaler_sha256", "y_scaler_sha256", "bundle_sha256"))


def test_three_fresh_models_and_no_fold_checkpoint_reuse(preflight):
    assert preflight["expected_new_runs"] == 3
    assert preflight["fresh_model_each_seed"] is True
    assert preflight["fresh_optimizer_each_seed"] is True
    assert preflight["scheduler"] == "OFF"
    assert preflight["validation_used"] is False
    assert preflight["early_stopping"] is False
    assert preflight["source_checkpoint_audit"]["source_count"] == 9
    assert preflight["source_checkpoint_audit"]["reuse_as_final_models"] is False
    # Step 16 was subsequently Human-authorized and completed. Preflight must
    # now verify/reuse the three immutable completed final-refit runs.
    assert preflight["run_ledger"]["completed_seeds"] == [42, 123, 2026]
    assert preflight["run_ledger"]["missing_seeds"] == []


def test_final_config_and_equal_weight_policy(preflight):
    assert set(preflight["final_refit_config_fingerprints"]) == {"42", "123", "2026"}
    assert len(set(preflight["final_refit_config_fingerprints"].values())) == 3
    assert preflight["candidate_id"] == "TR_C2_ALT_LOOKBACK_E14_M1"
    assert preflight["weights"] == [1 / 3, 1 / 3, 1 / 3]


def test_official_refuses_without_human_authorization(monkeypatch):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(step16, "run_training", forbidden)
    assert step16.main(["--experiment", "STEP16", "--mode", "official"]) == 3
    assert step16.main(["--experiment", "STEP16", "--mode", "resume-partial"]) == 3
    assert called is False


def test_preflight_never_accepts_training_authorization(monkeypatch):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(step16, "run_preflight", forbidden)
    assert step16.main(["--experiment", "STEP16", "--mode", "preflight", "--authorize-training"]) == 2
    assert called is False


def test_recovery_ledger_reuses_completed_and_preserves_failed(tmp_path, monkeypatch):
    scaler = SimpleNamespace(x_sha256="x", y_sha256="y")
    monkeypatch.setattr(step16, "final_refit_config", lambda _root, seed, _scaler: {"seed": seed})
    run_root = tmp_path / step16.ARTIFACT_ROOT / "runs"
    completed = run_root / "RUN_V2_TR_STEP16_SEED42_FINAL_A01_DEADBEEF"
    completed.joinpath("checkpoints").mkdir(parents=True)
    completed.joinpath("checkpoints/final_refit.pt").write_bytes(b"checkpoint")
    fingerprint = step16.compute_config_fingerprint({"seed": 42})
    completed.joinpath("config.json").write_text(json.dumps({"config_fingerprint": fingerprint}))
    with completed.joinpath("training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch"])
        writer.writeheader()
        writer.writerows({"epoch": value} for value in range(1, 17))
    completed.joinpath("status.json").write_text(json.dumps({
        "run_id": completed.name,
        "seed": 42,
        "status": "COMPLETED",
        "official_epoch": 16,
        "config_fingerprint": fingerprint,
        "checkpoint_sha256": step16.sha256_file(completed / "checkpoints/final_refit.pt"),
        "x_scaler_sha256": "x",
        "y_scaler_sha256": "y",
        "test_status": "NOT_ACCESSED",
    }))
    failed = run_root / "RUN_V2_TR_STEP16_SEED123_FINAL_A01_DEADBEEF"
    failed.mkdir()
    failed.joinpath("status.json").write_text(json.dumps({"seed": 123, "status": "FAILED"}))
    ledger = step16.audit_run_ledger(tmp_path, scaler)
    assert sorted(ledger["completed"]) == [42]
    assert ledger["missing_seeds"] == [123, 2026]
    assert ledger["failed_preserved"] == [failed.name]


def test_recovery_ledger_blocks_running_record(tmp_path):
    scaler = SimpleNamespace(x_sha256="x", y_sha256="y")
    running = tmp_path / step16.ARTIFACT_ROOT / "runs/RUN_V2_TR_STEP16_SEED42_FINAL_A01_DEADBEEF"
    running.mkdir(parents=True)
    running.joinpath("status.json").write_text(json.dumps({"seed": 42, "status": "RUNNING"}))
    with pytest.raises(step16.Step16PreflightError, match="require interruption audit"):
        step16.audit_run_ledger(tmp_path, scaler)
