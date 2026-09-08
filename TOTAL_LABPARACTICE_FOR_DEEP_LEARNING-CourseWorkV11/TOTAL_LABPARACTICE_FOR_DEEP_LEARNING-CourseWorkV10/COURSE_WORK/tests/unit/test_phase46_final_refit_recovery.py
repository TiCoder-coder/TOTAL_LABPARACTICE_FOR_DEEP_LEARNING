"""Phase 46 — FINAL_REFIT registry recovery tests.

Tests verify FINAL_REFIT-specific registry behavior:
  1. register_metric accepts FINAL_DEV fingerprint in FINAL_REFIT mode.
  2. register_metric rejects wrong fingerprints in FINAL_REFIT mode.
  3. recover_final_refit_run finalizes RUNNING run with FINAL_DEV metrics.
  4. recover_final_refit_run refuses non-FINAL_REFIT mode runs.
  5. recover_final_refit_run refuses non-RUNNING runs.
  6. complete_run accepts FINAL_REFIT runs with FINAL_DEV metrics.
  7. complete_run rejects incomplete FINAL_REFIT runs.
  8. RUN_TR_FSD_0181_2B11AC68 is COMPLETED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _sandbox_registry():
    from course_work.experiments.registry import ExperimentRegistry
    return ExperimentRegistry(ROOT)


def _register_test_run(final_refit: bool = True, seed: int = 99):
    reg = _sandbox_registry()
    cfg = {
        "data": {
            "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
            "feature_count": 33,
            "feature_variant_id": "FS2_TF1",
            "horizon_steps": 1,
            "lookback_steps": 72,
            "sampling_interval_minutes": 10,
            "target_access_mode": "FINAL_DEV",
            "target_scaling_option": "YS1",
            "test_sample_count": 0,
            "train_sample_count": 13670,
            "validation_sample_count": 2960,
        },
        "lineage": {
            "environment_id": "ENV-v1",
            "dataset_revision": "DATA-v1",
            "schema_version": "SCHEMA-v1",
            "temporal_version": "TEMPORAL-v1",
            "eda_version": "EDA-v1",
            "feature_version": "FEATURES-v1",
            "feature_set_version": "FEATURESETS-v1",
            "split_version": "SPLIT-v1",
            "scaling_version": "SCALING-v1",
            "window_version": "WINDOWS-v1",
            "population_version": "WINDOWPOP-v1",
            "dataloader_version": "DATALOADERS-v1",
            "metric_version": "METRICS-v1",
            "global_split_fingerprint": "4d0115b92a3c81406b62deb4d36dd8ceac847f948294cdc2e4dc49338ebeb821",
            "population_fingerprint": "a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987",
            "metric_contract_fingerprint": "4509825a7be2ee75220f88bedf31da6d062e6fee5168af45b0d54ec1db137198",
            "dataset_fingerprint": "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d",
            "final_dev_population_fingerprint": "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39",
            "final_scaling_x_sha256": "7280c166232ac53ef5947fa1991a1b38e9f5ec75045711b7092ddba9c53a17fd",
            "final_scaling_y_sha256": "e8c8edb970591afa5c25faf619d2257b544b1b27c0b725b3be376a52a5946cca",
            "final_dev_region_version": "FINAL_DEV_REGION-v1",
            "feature_fingerprint": "fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee",
            "scaler_bundle_id": "XSCALER__FS2_TF1",
            "scaler_bundle_checksum": "4e7c96d5accc2917855ed669b33d2666b3aa07f6b83b40097a9690c6a5f5e334",
            "target_scaler_bundle_id": "YSCALER__YS1",
            "target_scaler_checksum": "b3326a79da81b092460ef8d4a140a101b21f2433ff30c36971d626d2a2491697",
            "dataloader_fingerprint": "48dff0f10ac2165443f3e2ab60f93d3d1ce9149b70c985fbbe31f4059eb4cf7e",
            "window_fingerprint": "d4fc515a7f75f108a484e0d10cdf5259370af6181b6b5b88cdb1cdf501fa6bca",
        },
        "model": {
            "activation": "GELU",
            "attention_aware": True,
            "d_model": 64,
            "dropout": 0.1,
            "ffn_dim": 256,
            "implementation_version": "TRANSFORMER_IMPL-v1",
            "input_size": 33,
            "model_family": "TRANSFORMER_ENCODER",
            "model_name": "Transformer Encoder Regressor",
            "model_version": "TRANSFORMER-v1",
            "norm_first": False,
            "num_heads": 4,
            "num_layers": 2,
            "output_size": 1,
            "pooling": "LAST_STEP",
            "positional_encoding_type": "SINUSOIDAL",
        },
        "reproducibility": {"seed": seed},
        "runtime": {"device_name": "cpu", "device_type": "cpu", "dtype": "float32"},
        "training": {
            "batch_size": 32,
            "early_stopping_enabled": False,
            "early_stopping_metric": "rmse_wh",
            "early_stopping_mode": "MIN",
            "early_stopping_patience": 10,
            "enabled": True,
            "final_refit_mode": final_refit,
            "gradient_clip_max_norm": 1.0,
            "gradient_clipping_enabled": True,
            "learning_rate": 0.0003,
            "loss_name": "MSE",
            "max_epochs": 30 if final_refit else 50,
            "optimizer_name": "AdamW",
            "scheduler_config": None,
            "scheduler_name": None,
            "seed": seed,
            "weight_decay": 0.001,
        },
    }
    run_id = reg.register_run(cfg, "FINAL_SEED_RUN", "TRAINING", sweep_stage=f"SEED_{seed}", rerun_reason="PHASE46_CORRECTIVE_RERUN")["run_id"]
    reg.start_run(run_id)
    return run_id, reg


def _quarantine_run(run_id: str, reg) -> None:
    try:
        reg.fail_run(run_id, "OTHER", "test_teardown", "Test quarantine cleanup")
    except Exception:
        pass


def _materialize_test_run_artifacts(run_id: str) -> None:
    run_dir = ROOT / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)

    (run_dir / "checkpoints" / "best_checkpoint.pt").write_bytes(b"FAKE")
    (run_dir / "checkpoints" / "last_checkpoint.pt").write_bytes(b"FAKE")
    (run_dir / "training.log").write_text("best_epoch=30\nstopped_reason=MAX_EPOCHS\n")
    csv_rows = ["epoch,train_loss"]
    for i in range(1, 31):
        csv_rows.append(f"{i},0.5")
    (run_dir / "training_history.csv").write_text("\n".join(csv_rows))

    metric_payload = {
        "metric_result": {
            "horizon_steps": 1,
            "lookback_steps": 72,
            "mae_wh": 22.96,
            "metric_contract_fingerprint": "4509825a7be2ee75220f88bedf31da6d062e6fee5168af45b0d54ec1db137198",
            "metric_version": "METRICS-v1",
            "model_id": "Transformer Encoder Regressor",
            "n_samples": 16630,
            "population_fingerprint": "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39",
            "prediction_unit": "Wh",
            "r2": 0.789,
            "r2_status": "DEFINED",
            "rmse_wh": 47.855,
            "run_id": run_id,
            "selection_metric_name": "rmse_wh",
            "selection_metric_value": 47.855,
            "split_id": "FINAL_DEV",
            "status": "PASS",
            "target_unit": "Wh",
            "warnings": [],
        }
    }
    (run_dir / "metrics" / "best_validation_metrics.json").write_text(json.dumps(metric_payload))
    (run_dir / "predictions" / "best_validation_predictions.csv").write_text(
        "run_id,sample_idx,y_true_wh,y_pred_wh,residual_wh\n")

def test_register_metric_accepts_final_dev_fingerprint_in_final_refit_mode():
    reg = _sandbox_registry()
    run_id, _ = _register_test_run(final_refit=True, seed=991)
    _materialize_test_run_artifacts(run_id)
    try:
        fp = "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39"
        row = reg.register_metric(
            run_id, "FINAL_DEV", "rmse_wh", 47.855, "Wh", 16630, fp, "epoch_30",
        )
        assert row["split_id"] == "FINAL_DEV"
        assert row["population_fingerprint"] == fp
        assert row["metric_value"] == 47.855
    finally:
        _quarantine_run(run_id, reg)


def test_register_metric_rejects_wrong_fingerprint_in_final_refit_mode():
    reg = _sandbox_registry()
    run_id, _ = _register_test_run(final_refit=True, seed=992)
    try:
        with pytest.raises(ValueError, match="Metric population fingerprint mismatch"):
            reg.register_metric(
                run_id, "FINAL_DEV", "rmse_wh", 47.855, "Wh", 16630,
                "deadbeef" * 8, "epoch_30",
            )
    finally:
        _quarantine_run(run_id, reg)


def test_register_metric_rejects_wrong_n_samples_in_final_refit_mode():
    reg = _sandbox_registry()
    run_id, _ = _register_test_run(final_refit=True, seed=993)
    try:
        fp = "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39"
        with pytest.raises(ValueError, match="sample count mismatch"):
            reg.register_metric(
                run_id, "FINAL_DEV", "rmse_wh", 47.855, "Wh", 9999, fp, "epoch_30",
            )
    finally:
        _quarantine_run(run_id, reg)


def test_recover_final_refit_run_completes_with_zero_optimizer_steps():
    """recover_final_refit_run uses only on-disk evidence; no optimizer.step."""
    reg = _sandbox_registry()
    run_id, _ = _register_test_run(final_refit=True, seed=994)
    _materialize_test_run_artifacts(run_id)
    rec = reg.recover_final_refit_run(run_id, best_epoch=30)
    assert rec["status"] == "COMPLETED"
    assert rec["best_epoch"] == 30
    assert abs(rec["best_validation_rmse_wh"] - 47.855) < 1e-3
    metrics = rec["metrics"]
    assert len(metrics) == 3
    assert {m["split_id"] for m in metrics} == {"FINAL_DEV"}
    assert {m["metric_name"] for m in metrics} == {"mae_wh", "rmse_wh", "r2"}


def _cleanup_test_runs():
    """Quarantine all test-created RUN_TR_FSD_* runs."""
    reg = _sandbox_registry()
    for rec in reg._load_records():
        rid = rec.get("run_id", "")
        if rid.startswith("RUN_TR_FSD_") and rec.get("status") in ("RUNNING", "REGISTERED", "COMPLETED"):
            if rec.get("rerun_reason") in ("PHASE46_CORRECTIVE_RERUN", "PHASE46_UNIT_TEST"):
                try:
                    reg.fail_run(rid, "OTHER", "teardown", "Test quarantine cleanup")
                except Exception:
                    pass


class TestPhase46FinalRefitRecovery:
    @classmethod
    def setup_class(cls):
        _cleanup_test_runs()

    def teardown_method(self, method):
        pass

    def teardown_class(cls):
        _cleanup_test_runs()

    def test_register_metric_accepts_final_dev_fingerprint_in_final_refit_mode(self):
        reg = _sandbox_registry()
        run_id, _ = _register_test_run(final_refit=True, seed=991)
        _materialize_test_run_artifacts(run_id)
        try:
            fp = "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39"
            row = reg.register_metric(
                run_id, "FINAL_DEV", "rmse_wh", 47.855, "Wh", 16630, fp, "epoch_30",
            )
            assert row["split_id"] == "FINAL_DEV"
            assert row["population_fingerprint"] == fp
            assert row["metric_value"] == 47.855
        finally:
            _quarantine_run(run_id, reg)

    def test_register_metric_rejects_wrong_fingerprint_in_final_refit_mode(self):
        reg = _sandbox_registry()
        run_id, _ = _register_test_run(final_refit=True, seed=992)
        try:
            with pytest.raises(ValueError, match="Metric population fingerprint mismatch"):
                reg.register_metric(
                    run_id, "FINAL_DEV", "rmse_wh", 47.855, "Wh", 16630,
                    "deadbeef" * 8, "epoch_30",
                )
        finally:
            _quarantine_run(run_id, reg)

    def test_register_metric_rejects_wrong_n_samples_in_final_refit_mode(self):
        reg = _sandbox_registry()
        run_id, _ = _register_test_run(final_refit=True, seed=993)
        try:
            fp = "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39"
            with pytest.raises(ValueError, match="sample count mismatch"):
                reg.register_metric(
                    run_id, "FINAL_DEV", "rmse_wh", 47.855, "Wh", 9999, fp, "epoch_30",
                )
        finally:
            _quarantine_run(run_id, reg)

    def test_recover_final_refit_run_completes_with_zero_optimizer_steps(self):
        """recover_final_refit_run uses only on-disk evidence; no optimizer.step."""
        reg = _sandbox_registry()
        run_id, _ = _register_test_run(final_refit=True, seed=994)
        _materialize_test_run_artifacts(run_id)
        rec = reg.recover_final_refit_run(run_id, best_epoch=30)
        assert rec["status"] == "COMPLETED"
        assert rec["best_epoch"] == 30
        assert abs(rec["best_validation_rmse_wh"] - 47.855) < 1e-3
        metrics = rec["metrics"]
        assert len(metrics) == 3
        assert {m["split_id"] for m in metrics} == {"FINAL_DEV"}
        assert {m["metric_name"] for m in metrics} == {"mae_wh", "rmse_wh", "r2"}

    def test_recover_final_refit_run_refuses_non_running_status(self):
        reg = _sandbox_registry()
        run_id, _ = _register_test_run(final_refit=True, seed=995)
        _materialize_test_run_artifacts(run_id)
        reg.fail_run(run_id, "OTHER", "setup", "test")
        with pytest.raises(ValueError, match="only RUNNING runs"):
            reg.recover_final_refit_run(run_id, best_epoch=30)

    def test_recover_final_refit_run_refuses_non_final_refit_mode(self):
        reg = _sandbox_registry()
        run_id, _ = _register_test_run(final_refit=False, seed=996)
        try:
            with pytest.raises(ValueError, match="not in FINAL_REFIT mode"):
                reg.recover_final_refit_run(run_id, best_epoch=50)
        finally:
            _quarantine_run(run_id, reg)

    def test_complete_run_accepts_final_refit_with_final_dev_metrics(self):
        """complete_run succeeds when FINAL_DEV metrics are registered."""
        reg = _sandbox_registry()
        run_id, _ = _register_test_run(final_refit=True, seed=997)
        _materialize_test_run_artifacts(run_id)
        fp = "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39"
        reg.register_metric(run_id, "FINAL_DEV", "mae_wh", 22.96, "Wh", 16630, fp, "epoch_30")
        reg.register_metric(run_id, "FINAL_DEV", "rmse_wh", 47.855, "Wh", 16630, fp, "epoch_30")
        reg.register_metric(run_id, "FINAL_DEV", "r2", 0.789, "dimensionless", 16630, fp, "epoch_30")
        rec = reg.complete_run(run_id, best_epoch=30)
        assert rec["status"] == "COMPLETED"
        assert rec["best_epoch"] == 30
        assert abs(rec["best_validation_rmse_wh"] - 47.855) < 1e-3

    def test_complete_run_rejects_incomplete_final_refit_metrics(self):
        """complete_run fails when mae_wh is missing."""
        reg = _sandbox_registry()
        run_id, _ = _register_test_run(final_refit=True, seed=998)
        _materialize_test_run_artifacts(run_id)
        fp = "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39"
        reg.register_metric(run_id, "FINAL_DEV", "rmse_wh", 47.855, "Wh", 16630, fp, "epoch_30")
        with pytest.raises(ValueError, match="FINAL_DEV metrics are incomplete"):
            reg.complete_run(run_id, best_epoch=30)


def test_seed42_run_is_completed():
    """Verify RUN_TR_FSD_0181_2B11AC68 is COMPLETED in the real registry."""
    reg = _sandbox_registry()
    rec = reg.get_run("RUN_TR_FSD_0181_2B11AC68")
    assert rec["status"] == "COMPLETED"
    assert rec["best_epoch"] == 30
    assert rec["best_validation_rmse_wh"] == pytest.approx(47.8557, rel=1e-4)
    metric_names = {m["metric_name"] for m in rec["metrics"]}
    assert metric_names == {"mae_wh", "rmse_wh", "r2"}
    splits = {m["split_id"] for m in rec["metrics"]}
    assert splits == {"FINAL_DEV"}

