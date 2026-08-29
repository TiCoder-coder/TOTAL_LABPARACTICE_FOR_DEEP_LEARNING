import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from course_work.attention.verification import materialize_phase_17
from course_work.baselines.lstm_baseline import _failure_type_for_stage, _safe_failure_message
from course_work.experiments.registry import (
    ArtifactType,
    ExecutionType,
    ExperimentRegistry,
    FailureType,
    RunStatus,
    build_reference_run_config,
    canonicalize_config,
    compare_configs,
    compute_config_fingerprint,
    validate_run_config,
    validate_status_transition,
    validate_sweep_consistency,
)
from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes


class ExperimentRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.base_config = build_reference_run_config(cls.root)
        cls.persistence_config = build_reference_run_config(cls.root, "PERSISTENCE")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="registry-test-")
        self.temporary_root = Path(self.temporary.name)
        self.registry = ExperimentRegistry(
            self.root,
            self.temporary_root / "registry",
            self.temporary_root / "runs",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def register(self, execution_type: str = ExecutionType.SANITY.value) -> dict:
        return self.registry.register_run(
            deepcopy(self.base_config),
            "TRANSFORMER_BASELINE",
            execution_type,
        )

    def artifact(self, run_id: str, name: str, content: str = "TEST_ONLY") -> Path:
        path = self.temporary_root / "runs" / run_id / name
        atomic_write_bytes(path, content.encode("utf-8"))
        return path

    def register_validation_metrics(self, run_id: str, config: dict | None = None) -> None:
        selected = config or self.base_config
        population = selected["lineage"]["population_fingerprint"]
        samples = selected["data"]["validation_sample_count"]
        for name, value, unit in (
            ("mae_wh", 8.0, "Wh"),
            ("rmse_wh", 10.0, "Wh"),
            ("r2", 0.8, "dimensionless"),
        ):
            self.registry.register_metric(run_id, "VALIDATION", name, value, unit, samples, population, "BEST")

    def test_canonical_config_normalizes_float_enum_and_integer_fields(self) -> None:
        equivalent = deepcopy(self.base_config)
        equivalent["model"]["activation"] = "gelu"
        equivalent["training"]["learning_rate"] = 0.0003
        equivalent["data"]["lookback_steps"] = 144.0
        self.assertEqual(compute_config_fingerprint(self.base_config), compute_config_fingerprint(equivalent))
        self.assertEqual(canonicalize_config(equivalent)["model"]["activation"], "GELU")
        self.assertEqual(canonicalize_config(equivalent)["data"]["lookback_steps"], 144)

    def test_result_and_timestamp_fields_do_not_change_config_fingerprint(self) -> None:
        changed = deepcopy(self.base_config)
        changed["created_at"] = "TEST_ONLY"
        changed["best_epoch"] = 999
        self.assertEqual(compute_config_fingerprint(self.base_config), compute_config_fingerprint(changed))

    def test_semantic_validation_rejects_invalid_transformer_and_data_contracts(self) -> None:
        invalid_heads = deepcopy(self.base_config)
        invalid_heads["model"]["num_heads"] = 3
        with self.assertRaises(ValueError):
            validate_run_config(invalid_heads, self.registry.upstream_context)
        invalid_lookback = deepcopy(self.base_config)
        invalid_lookback["data"]["lookback_steps"] = 100
        with self.assertRaises(ValueError):
            validate_run_config(invalid_lookback, self.registry.upstream_context)
        invalid_lineage = deepcopy(self.base_config)
        invalid_lineage["lineage"]["split_version"] = "SPLIT-v2"
        with self.assertRaises(ValueError):
            validate_run_config(invalid_lineage, self.registry.upstream_context)

    def test_persistence_config_uses_task_level_null_dependencies(self) -> None:
        config = validate_run_config(deepcopy(self.persistence_config), self.registry.upstream_context)
        self.assertEqual(config["model"]["model_name"], "PERSISTENCE_LAST_VALUE")
        self.assertEqual(config["model"]["baseline_scope"], "TASK_LEVEL")
        self.assertEqual(config["model"]["trainable_parameters"], 0)
        self.assertIsNone(config["data"]["feature_variant_id"])
        self.assertIsNone(config["data"]["target_scaling_option"])
        self.assertIsNone(config["lineage"]["scaler_bundle_id"])
        self.assertIsNone(config["lineage"]["dataloader_fingerprint"])
        self.assertIsNone(config["reproducibility"]["seed"])
        self.assertFalse(config["training"]["enabled"])
        self.assertEqual(config["runtime"]["device_type"], "cpu")

    def test_persistence_config_rejects_neural_dependencies(self) -> None:
        invalid_feature = deepcopy(self.persistence_config)
        invalid_feature["data"]["feature_variant_id"] = "FS1_TF1"
        with self.assertRaises(ValueError):
            validate_run_config(invalid_feature, self.registry.upstream_context)
        invalid_scaler = deepcopy(self.persistence_config)
        invalid_scaler["lineage"]["scaler_bundle_id"] = "SCALER_TEST_ONLY"
        with self.assertRaises(ValueError):
            validate_run_config(invalid_scaler, self.registry.upstream_context)
        invalid_seed = deepcopy(self.persistence_config)
        invalid_seed["reproducibility"]["seed"] = 42
        with self.assertRaises(ValueError):
            validate_run_config(invalid_seed, self.registry.upstream_context)
        invalid_training = deepcopy(self.persistence_config)
        invalid_training["training"]["loss_name"] = "MSE"
        with self.assertRaises(ValueError):
            validate_run_config(invalid_training, self.registry.upstream_context)

    def test_run_ids_are_unique_and_duplicate_config_requires_reason(self) -> None:
        first = self.register()
        with self.assertRaises(ValueError):
            self.register()
        second = self.registry.register_run(
            deepcopy(self.base_config),
            "TRANSFORMER_BASELINE",
            ExecutionType.SANITY.value,
            parent_run_id=first["run_id"],
            rerun_reason="REPRODUCIBILITY_CHECK",
        )
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["config_fingerprint"], second["config_fingerprint"])

    def test_parent_run_must_exist(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.register_run(
                deepcopy(self.base_config),
                "TRANSFORMER_BASELINE",
                ExecutionType.SANITY.value,
                parent_run_id="MISSING",
            )

    def test_status_machine_rejects_completed_to_running(self) -> None:
        validate_status_transition(RunStatus.REGISTERED.value, RunStatus.RUNNING.value)
        with self.assertRaises(ValueError):
            validate_status_transition(RunStatus.COMPLETED.value, RunStatus.RUNNING.value)

    def test_sanity_lifecycle_requires_audit_and_completed_config_is_immutable(self) -> None:
        run = self.register()
        self.registry.start_run(run["run_id"])
        with self.assertRaises(ValueError):
            self.registry.complete_run(run["run_id"])
        audit = self.artifact(run["run_id"], "audit.json")
        self.registry.register_artifact(run["run_id"], ArtifactType.AUDIT.value, audit, True)
        completed = self.registry.complete_run(run["run_id"])
        self.assertEqual(completed["status"], RunStatus.COMPLETED.value)
        config_path = self.temporary_root / "runs" / run["run_id"] / "config.json"
        payload = json.loads(config_path.read_text())
        payload["config"]["training"]["learning_rate"] = 0.5
        atomic_write_bytes(config_path, canonical_json_bytes(payload))
        with self.assertRaises(RuntimeError):
            self.registry.get_run(run["run_id"])

    def test_failed_run_requires_taxonomy_stage_and_message(self) -> None:
        run = self.register()
        self.registry.start_run(run["run_id"])
        with self.assertRaises(ValueError):
            self.registry.fail_run(run["run_id"], FailureType.NUMERICAL_ERROR.value, "", "")
        failed = self.registry.fail_run(run["run_id"], FailureType.NUMERICAL_ERROR.value, "TRAINING", "TEST_ONLY")
        self.assertEqual(failed["status"], RunStatus.FAILED.value)
        self.assertEqual(failed["failure"]["failure_type"], FailureType.NUMERICAL_ERROR.value)

    def test_baseline_interrupt_uses_safe_failure_message(self) -> None:
        self.assertEqual(_safe_failure_message(KeyboardInterrupt()), "KeyboardInterrupt")
        self.assertEqual(_failure_type_for_stage("TRAIN", KeyboardInterrupt()), FailureType.INTERRUPTED.value)

    def test_phase_17_manifest_includes_reference_model_version(self) -> None:
        materialize_phase_17(self.root)
        manifest = (self.root / "artifacts/attention_verification/attention_verification_manifest.json").read_text()
        self.assertIn('"reference_model_version"', manifest)
        self.assertIn('"unit_test_count"', manifest)

    def test_development_test_target_and_metric_are_rejected(self) -> None:
        invalid = deepcopy(self.base_config)
        invalid["data"]["target_access_mode"] = "TEST"
        with self.assertRaises(PermissionError):
            self.registry.register_run(invalid, "TRANSFORMER_BASELINE", ExecutionType.TRAINING.value)
        run = self.registry.register_run(deepcopy(self.base_config), "TRANSFORMER_BASELINE", ExecutionType.TRAINING.value)
        self.registry.start_run(run["run_id"])
        with self.assertRaises(PermissionError):
            self.registry.register_metric(
                run["run_id"],
                "TEST",
                "rmse_wh",
                10.0,
                "Wh",
                self.base_config["data"]["test_sample_count"],
                self.base_config["lineage"]["population_fingerprint"],
                "BEST",
            )

    def test_final_test_requires_lock_and_authorization_then_accepts_test_metrics(self) -> None:
        config = deepcopy(self.base_config)
        config["data"]["target_access_mode"] = "TEST"
        with self.assertRaises(PermissionError):
            self.registry.register_run(config, "FINAL_TEST", ExecutionType.FINAL_TEST.value)
        run = self.registry.register_run(
            config,
            "FINAL_TEST",
            ExecutionType.FINAL_TEST.value,
            final_model_lock_id="TEST_ONLY_LOCK",
            test_access_authorized=True,
        )
        self.registry.start_run(run["run_id"])
        population = config["lineage"]["population_fingerprint"]
        samples = config["data"]["test_sample_count"]
        metric = self.registry.register_metric(run["run_id"], "TEST", "rmse_wh", 10.0, "Wh", samples, population, "FINAL")
        self.assertEqual(metric["split_id"], "TEST")

    def test_metric_registration_guards_unit_population_count_and_duplicates(self) -> None:
        run = self.registry.register_run(deepcopy(self.base_config), "TRANSFORMER_BASELINE", ExecutionType.TRAINING.value)
        self.registry.start_run(run["run_id"])
        population = self.base_config["lineage"]["population_fingerprint"]
        samples = self.base_config["data"]["validation_sample_count"]
        with self.assertRaises(ValueError):
            self.registry.register_metric(run["run_id"], "VALIDATION", "rmse_wh", 1.0, "dimensionless", samples, population, "BEST")
        with self.assertRaises(ValueError):
            self.registry.register_metric(run["run_id"], "VALIDATION", "rmse_wh", 1.0, "Wh", samples, "other", "BEST")
        with self.assertRaises(ValueError):
            self.registry.register_metric(run["run_id"], "VALIDATION", "rmse_wh", 1.0, "Wh", samples - 1, population, "BEST")
        self.registry.register_metric(run["run_id"], "VALIDATION", "rmse_wh", 1.0, "Wh", samples, population, "BEST")
        with self.assertRaises(ValueError):
            self.registry.register_metric(run["run_id"], "VALIDATION", "rmse_wh", 1.0, "Wh", samples, population, "BEST")

    def test_final_refit_mode_allows_combined_pretest_population_metrics(self) -> None:
        config = deepcopy(self.base_config)
        config["training"]["final_refit_mode"] = True
        run = self.registry.register_run(config, "TRANSFORMER_BASELINE", ExecutionType.TRAINING.value)
        self.registry.start_run(run["run_id"])
        population = config["lineage"]["population_fingerprint"]
        train_samples = config["data"]["train_sample_count"]
        for kind, name in (
            (ArtifactType.TRAIN_LOG.value, "training.csv"),
            (ArtifactType.BEST_CHECKPOINT.value, "best.pt"),
            (ArtifactType.METRICS.value, "metrics.json"),
        ):
            self.registry.register_artifact(run["run_id"], kind, self.artifact(run["run_id"], name), True)
        self.registry.register_metric(run["run_id"], "VALIDATION", "mae_wh", 8.0, "Wh", train_samples, population, "FINAL_REFIT")
        self.registry.register_metric(run["run_id"], "VALIDATION", "rmse_wh", 10.0, "Wh", train_samples, population, "FINAL_REFIT")
        self.registry.register_metric(run["run_id"], "VALIDATION", "r2", 0.8, "dimensionless", train_samples, population, "FINAL_REFIT")
        completed = self.registry.complete_run(run["run_id"], best_epoch=7, best_validation_rmse_wh=10.0)
        self.assertEqual(completed["status"], RunStatus.COMPLETED.value)

    def test_training_completion_requires_artifacts_metrics_and_matching_best_value(self) -> None:
        run = self.registry.register_run(deepcopy(self.base_config), "TRANSFORMER_BASELINE", ExecutionType.TRAINING.value)
        self.registry.start_run(run["run_id"])
        for kind, name in (
            (ArtifactType.TRAIN_LOG.value, "training.csv"),
            (ArtifactType.BEST_CHECKPOINT.value, "best.pt"),
            (ArtifactType.METRICS.value, "metrics.json"),
        ):
            self.registry.register_artifact(run["run_id"], kind, self.artifact(run["run_id"], name), True)
        self.register_validation_metrics(run["run_id"])
        with self.assertRaises(ValueError):
            self.registry.complete_run(run["run_id"], best_epoch=2, best_validation_rmse_wh=11.0)
        completed = self.registry.complete_run(run["run_id"], best_epoch=2, best_validation_rmse_wh=10.0)
        self.assertEqual(completed["best_epoch"], 2)
        self.assertEqual(completed["best_validation_rmse_wh"], 10.0)

    def test_persistence_evaluation_requires_predictions_and_metrics(self) -> None:
        run = self.registry.register_run(
            deepcopy(self.persistence_config),
            "PERSISTENCE_BASELINE",
            ExecutionType.EVALUATION.value,
        )
        self.registry.start_run(run["run_id"])
        metrics = self.artifact(run["run_id"], "metrics.json")
        self.registry.register_artifact(run["run_id"], ArtifactType.METRICS.value, metrics, True)
        self.register_validation_metrics(run["run_id"], self.persistence_config)
        with self.assertRaises(ValueError):
            self.registry.complete_run(run["run_id"])
        predictions = self.artifact(run["run_id"], "predictions.csv")
        self.registry.register_artifact(run["run_id"], ArtifactType.PREDICTIONS.value, predictions, True)
        completed = self.registry.complete_run(run["run_id"])
        self.assertEqual(completed["status"], RunStatus.COMPLETED.value)
        self.assertEqual(completed["best_validation_rmse_wh"], 10.0)

    def test_sweep_consistency_accepts_one_factor_and_rejects_confounding(self) -> None:
        learning_rate = deepcopy(self.base_config)
        learning_rate["training"]["learning_rate"] = 1e-3
        result = validate_sweep_consistency([self.base_config, learning_rate], {"training.learning_rate"})
        self.assertEqual(result["status"], "PASS")
        learning_rate["model"]["dropout"] = 0.2
        with self.assertRaises(ValueError):
            validate_sweep_consistency([self.base_config, learning_rate], {"training.learning_rate"})

    def test_sweep_registration_is_persisted_and_run_reference_is_guarded(self) -> None:
        sweep = self.registry.register_sweep(
            "S8_LR_TEST_ONLY",
            "S8",
            "S8_LEARNING_RATE",
            "training.learning_rate",
            [3e-4, 1e-3],
            deepcopy(self.base_config),
        )
        self.assertEqual(sweep["status"], RunStatus.PLANNED.value)
        run = self.registry.register_run(
            deepcopy(self.base_config),
            "S8_LEARNING_RATE",
            ExecutionType.TRAINING.value,
            sweep_id=sweep["sweep_id"],
            sweep_stage="S8",
        )
        self.assertEqual(run["sweep_id"], sweep["sweep_id"])
        self.assertEqual(len(self.registry.get_runs_by_sweep(sweep["sweep_id"])), 1)
        with self.assertRaises(ValueError):
            self.registry.register_run(
                deepcopy(self.base_config),
                "S8_LEARNING_RATE",
                ExecutionType.TRAINING.value,
                sweep_id="UNKNOWN",
            )

    def test_comparison_reports_expected_and_unexpected_differences(self) -> None:
        candidate = deepcopy(self.base_config)
        candidate["training"]["learning_rate"] = 1e-3
        expected = compare_configs(self.base_config, candidate, {"training.learning_rate"})
        self.assertEqual(expected["status"], "COMPARABLE_WITH_EXPECTED_DIFFERENCES")
        unexpected = compare_configs(self.base_config, candidate, set())
        self.assertEqual(unexpected["status"], "NOT_COMPARABLE")

    def test_registry_validation_detects_artifact_checksum_change(self) -> None:
        run = self.register()
        self.registry.start_run(run["run_id"])
        audit = self.artifact(run["run_id"], "audit.json")
        self.registry.register_artifact(run["run_id"], ArtifactType.AUDIT.value, audit, True)
        self.registry.complete_run(run["run_id"])
        self.assertTrue(all(row["status"] == "PASS" for row in self.registry.validate_registry()))
        atomic_write_bytes(audit, b"CHANGED")
        with self.assertRaises(RuntimeError):
            self.registry.validate_registry()


if __name__ == "__main__":
    unittest.main()
