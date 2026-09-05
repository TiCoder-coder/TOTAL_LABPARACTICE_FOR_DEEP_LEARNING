import csv
import json
import unittest
from pathlib import Path

from course_work.baselines.persistence import materialize_phase_14
from course_work.contracts.coursework import materialize_phase_0
from course_work.data.acquisition import materialize_phase_2
from course_work.data.datasets import (
    DATALOADER_VERSION,
    build_dataset_suite,
    materialize_phase_11,
)
from course_work.data.feature_sets import load_validated_feature_set_registry, materialize_phase_8
from course_work.data.features import ENGINEERED_FEATURES, load_validated_feature_view, materialize_phase_7
from course_work.data.schema import materialize_phase_3
from course_work.data.scaling import (
    SCALING_VERSION,
    load_validated_scaler_bundle,
    load_validated_target_scaler,
    materialize_phase_9,
)
from course_work.data.splitting import load_validated_split_membership, materialize_phase_5
from course_work.data.temporal import materialize_phase_4
from course_work.data.windows import (
    POPULATION_VERSION,
    WINDOW_VERSION,
    load_validated_common_population,
    load_validated_window_index,
    materialize_phase_10,
)
from course_work.evaluation.metrics import METRIC_VERSION, materialize_phase_12
from course_work.experiments.registry import EXPERIMENT_VERSION, materialize_phase_13
from course_work.reporting.eda import FIGURE_FILENAMES, materialize_phase_6
from course_work.utils.artifacts import read_json, sha256_file
from course_work.utils.environment import materialize_phase_1


class PhaseChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_phase_chain_signoffs_are_valid(self) -> None:
        signoffs = [
            materialize_phase_0(self.root),
            materialize_phase_1(self.root),
            materialize_phase_2(self.root),
            materialize_phase_3(self.root),
            materialize_phase_4(self.root),
            materialize_phase_5(self.root),
            materialize_phase_6(self.root),
            materialize_phase_7(self.root),
            materialize_phase_8(self.root),
            materialize_phase_9(self.root),
            materialize_phase_10(self.root),
            materialize_phase_11(self.root),
            materialize_phase_12(self.root),
            materialize_phase_13(self.root),
            materialize_phase_14(self.root),
        ]
        self.assertEqual([item["phase_id"] for item in signoffs], list(range(15)))
        self.assertEqual([item["status"] for item in signoffs], ["PASS", "PASS", "PASS", "PASS_WITH_WARNING", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS"])

    def test_every_signed_output_checksum_matches(self) -> None:
        signoff_paths = [
            "artifacts/contracts/phase_0_signoff.json",
            "artifacts/environment/phase_1_signoff.json",
            "artifacts/acquisition/phase_2_signoff.json",
            "artifacts/schema/phase_3_signoff.json",
            "artifacts/temporal/phase_4_signoff.json",
            "artifacts/splits/phase_5_signoff.json",
            "artifacts/eda/phase_6_signoff.json",
            "artifacts/features/phase_7_signoff.json",
            "artifacts/feature_sets/phase_8_signoff.json",
            "artifacts/scaling/phase_9_signoff.json",
            "artifacts/windows/phase_10_signoff.json",
            "artifacts/dataloaders/phase_11_signoff.json",
            "artifacts/metrics/phase_12_signoff.json",
            "artifacts/experiments/phase_13_signoff.json",
            "artifacts/baselines/persistence/phase_14_signoff.json",
        ]
        for relative_signoff in signoff_paths:
            signoff = read_json(self.root / relative_signoff)
            for relative_output, expected_checksum in signoff["output_checksums"].items():
                self.assertEqual(sha256_file(self.root / relative_output), expected_checksum)

    def test_raw_data_is_unchanged(self) -> None:
        raw = self.root / "data/raw_data/energydata_complete.csv"
        self.assertEqual(sha256_file(raw), "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d")

    def test_eda_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/eda/eda_manifest.json")
        figures = self.root / "artifacts/eda/figures"
        self.assertEqual({path.name for path in figures.glob("*.png")}, set(FIGURE_FILENAMES))
        self.assertEqual(manifest["split_distribution_analysis_status"], "DERIVED_FROM_PHASE_5")
        self.assertEqual(manifest["processing_actions"]["rows_removed"], 0)
        self.assertFalse(manifest["processing_actions"]["outliers_removed"])
        self.assertFalse(manifest["processing_actions"]["interpolation_applied"])
        self.assertFalse(manifest["processing_actions"]["feature_selection_applied"])
        self.assertFalse(manifest["processing_actions"]["model_tuning_applied"])

    def test_phase_7_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/features/feature_engineering_manifest.json")
        view = load_validated_feature_view(self.root)
        audit_path = self.root / "artifacts/features/feature_engineering_audit.csv"
        self.assertEqual(manifest["feature_version"], "FEATURES-v1")
        self.assertEqual(manifest["engineered_features"], ENGINEERED_FEATURES)
        self.assertEqual(manifest["row_count"], len(view))
        self.assertEqual(manifest["column_count"], len(view.columns))
        self.assertEqual(manifest["audit_status"], "PASS")
        self.assertEqual(manifest["leakage_audit_status"], "PASS")
        self.assertTrue(audit_path.is_file())
        self.assertFalse(any(token in column for column in view.columns for token in ["_lag_", "rolling_", "target_next", "scaled_"]))
        self.assertEqual(sha256_file(self.root / manifest["derived_file_path"]), manifest["derived_file_sha256"])

    def test_phase_8_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/feature_sets/feature_set_manifest.json")
        registry = load_validated_feature_set_registry(self.root)
        self.assertEqual(manifest["feature_set_version"], "FEATURESETS-v1")
        self.assertEqual(manifest["feature_version"], "FEATURES-v1")
        self.assertEqual(manifest["baseline_variant"], "FS1_TF1")
        self.assertEqual(manifest["variant_feature_counts"], {
            "FS0_TF0": 25,
            "FS0_TF1": 30,
            "FS1_TF0": 26,
            "FS1_TF1": 31,
            "FS2_TF0": 28,
            "FS2_TF1": 33,
        })
        self.assertEqual(set(registry["variants"]), set(manifest["variant_ids"]))
        self.assertTrue(manifest["all_variants_valid"])
        self.assertTrue(manifest["leakage_audit_passed"])
        self.assertTrue(manifest["order_audit_passed"])

    def test_phase_5_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/splits/split_manifest.json")
        membership = load_validated_split_membership(self.root)
        distribution = self.root / "artifacts/splits/train_validation_distribution_summary.csv"
        self.assertEqual(manifest["split_version"], "SPLIT-v1")
        self.assertEqual(manifest["target_assignment_policy"], "by_target_timestamp")
        self.assertEqual(manifest["primary_boundary_protocol"], "WB0_CONTEXT_CARRY_OVER")
        self.assertEqual(manifest["test_distribution_analysis_status"], "LOCKED_UNTIL_PHASE_47")
        self.assertTrue(manifest["test_locked"])
        self.assertEqual(membership["split_id"].value_counts().to_dict(), {
            "TRAIN": 13814,
            "TEST": 2961,
            "VALIDATION": 2960,
        })
        self.assertTrue(distribution.is_file())
        self.assertNotIn("TEST", distribution.read_text(encoding="utf-8"))

    def test_phase_9_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/scaling/scaling_manifest.json")
        self.assertEqual(manifest["scaling_version"], SCALING_VERSION)
        self.assertEqual(manifest["x_fit_split"], "TRAIN")
        self.assertEqual(manifest["y_fit_split"], "TRAIN")
        self.assertEqual(manifest["train_row_count"], 13814)
        self.assertEqual(len(manifest["variant_scaler_bundles"]), 6)
        self.assertEqual(manifest["target_options"], ["YS0", "YS1"])
        self.assertTrue(manifest["leakage_audit_passed"])
        self.assertEqual(manifest["test_distribution_inspection"], "STRUCTURAL_ONLY")
        self.assertFalse(manifest["full_scaled_dataset_saved"])
        self.assertEqual(load_validated_scaler_bundle("FS1_TF1", self.root)["fit_split"], "TRAIN")
        self.assertEqual(load_validated_target_scaler(self.root)["fit_split"], "TRAIN")

    def test_phase_10_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/windows/window_manifest.json")
        index = load_validated_window_index(self.root)
        population = load_validated_common_population(self.root)
        self.assertEqual(manifest["window_version"], WINDOW_VERSION)
        self.assertEqual(manifest["population_version"], POPULATION_VERSION)
        self.assertEqual(manifest["lookback_options"], [36, 72, 144])
        self.assertEqual(manifest["forecast_horizon_steps"], 1)
        self.assertEqual(manifest["primary_boundary_protocol"], "WB0_CONTEXT_CARRY_OVER")
        self.assertEqual(manifest["test_target_access_policy"], "LOCKED_UNTIL_PHASE_47")
        self.assertFalse(manifest["test_target_values_exported"])
        self.assertFalse(manifest["full_3d_windows_saved"])
        self.assertEqual(population["target_split_id"].value_counts().to_dict(), {
            "TRAIN": 13670,
            "TEST": 2961,
            "VALIDATION": 2960,
        })
        self.assertEqual(index["lookback_steps"].value_counts().to_dict(), {
            36: 19591,
            72: 19591,
            144: 19591,
        })
        self.assertFalse(any("target_value" in column or "y_raw" in column for column in index.columns))
        expected_targets = set(population["target_sample_id"])
        for lookback in (36, 72, 144):
            actual_targets = set(index.loc[index["lookback_steps"].eq(lookback), "target_sample_id"])
            self.assertEqual(actual_targets, expected_targets)

    def test_phase_11_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/dataloaders/dataloader_manifest.json")
        datasets = build_dataset_suite(self.root)
        self.assertEqual(manifest["dataloader_version"], DATALOADER_VERSION)
        self.assertEqual(manifest["dataset_class"], "SequenceWindowDataset")
        self.assertTrue(manifest["lazy_materialization"])
        self.assertEqual(manifest["supported_batch_sizes"], [32, 64])
        self.assertEqual(manifest["baseline_batch_size"], 64)
        self.assertTrue(manifest["train_shuffle"])
        self.assertFalse(manifest["validation_shuffle"])
        self.assertFalse(manifest["test_shuffle"])
        self.assertFalse(manifest["drop_last"])
        self.assertEqual(manifest["baseline_num_workers"], 0)
        self.assertTrue(manifest["sample_coverage_passed"])
        self.assertTrue(manifest["shuffle_reproducibility_passed"])
        self.assertTrue(manifest["sequential_order_passed"])
        self.assertTrue(manifest["batch_shape_audit_passed"])
        self.assertTrue(manifest["test_firewall_passed"])
        self.assertEqual({split_id: len(dataset) for split_id, dataset in datasets.items()}, {
            "TRAIN": 13670,
            "VALIDATION": 2960,
            "TEST": 2961,
        })
        self.assertEqual(set(datasets["TEST"][0]), {"x", "sample_idx"})

    def test_phase_12_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/metrics/metric_manifest.json")
        contract = read_json(self.root / "artifacts/metrics/metric_contract.json")
        self.assertEqual(manifest["metric_version"], METRIC_VERSION)
        self.assertEqual(manifest["required_metrics"], ["mae_wh", "rmse_wh", "r2"])
        self.assertEqual(manifest["primary_selection_metric"], "rmse_wh")
        self.assertEqual(manifest["primary_selection_split"], "VALIDATION")
        self.assertEqual(manifest["target_unit"], "Wh")
        self.assertEqual(manifest["aggregation_policy"], "FULL_SPLIT_CONCATENATE_THEN_COMPUTE_ONCE")
        self.assertFalse(manifest["test_metrics_computed_in_phase12"])
        self.assertFalse(manifest["test_targets_materialized_in_phase12"])
        self.assertFalse(contract["r2_policy"]["force_finite"])
        self.assertEqual(contract["residual_definition"], "y_true_wh_minus_y_pred_wh")

    def test_phase_13_outputs_satisfy_boundary(self) -> None:
        manifest = read_json(self.root / "artifacts/experiments/registry_manifest.json")
        signoff = read_json(self.root / "artifacts/experiments/phase_13_signoff.json")
        with (self.root / "artifacts/experiments/experiment_families.csv").open(newline="", encoding="utf-8") as stream:
            families = list(csv.DictReader(stream))
        with (self.root / "artifacts/experiments/registry_validation_audit.csv").open(newline="", encoding="utf-8") as stream:
            audits = list(csv.DictReader(stream))
        registry_records = [
            json.loads(line)
            for line in (self.root / "artifacts/experiments/experiment_registry.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        run_ids = [record["run_id"] for record in registry_records]
        persistence_records = [
            record
            for record in registry_records
            if record["experiment_family"] == "PERSISTENCE_BASELINE"
        ]
        self.assertEqual(manifest["experiment_registry_version"], EXPERIMENT_VERSION)
        self.assertEqual(manifest["run_count"], len(registry_records))
        self.assertEqual(manifest["completed_count"], sum(record["status"] == "COMPLETED" for record in registry_records))
        self.assertEqual(manifest["active_count"], 0)
        self.assertEqual(manifest["failed_count"], 0)
        self.assertEqual(manifest["family_count"], 26)
        self.assertEqual(manifest["sweep_count"], 0)
        self.assertFalse(manifest["production_registry_empty"])
        self.assertFalse(manifest["synthetic_records_persisted"])
        self.assertTrue(manifest["test_firewall_enabled"])
        self.assertTrue(manifest["sweep_consistency_guard_enabled"])
        self.assertTrue(manifest["completed_run_immutability"])
        self.assertEqual(len(families), 26)
        self.assertTrue(audits)
        self.assertTrue(all(row["status"] == "PASS" for row in audits))
        self.assertEqual(len(run_ids), len(set(run_ids)))
        self.assertEqual(len(persistence_records), 1)
        self.assertEqual(persistence_records[0]["run_id"], "RUN_PS_PS_0001_AFFD3E3F")
        self.assertEqual(persistence_records[0]["status"], "COMPLETED")
        self.assertGreater((self.root / "artifacts/experiments/experiment_registry.jsonl").stat().st_size, 0)
        self.assertTrue((self.root / "artifacts/runs/RUN_PS_PS_0001_AFFD3E3F").is_dir())
        self.assertEqual(signoff["production_run_count"], 0)
        self.assertFalse(signoff["synthetic_records_persisted"])

    def test_phase_14_outputs_satisfy_boundary(self) -> None:
        signoff = read_json(self.root / "artifacts/baselines/persistence/phase_14_signoff.json")
        manifest = read_json(self.root / "artifacts/baselines/persistence/persistence_manifest.json")
        metrics = read_json(self.root / "artifacts/baselines/persistence/persistence_validation_metrics.json")
        with (self.root / "artifacts/baselines/persistence/persistence_validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        with (self.root / "artifacts/experiments/experiment_registry.csv").open(newline="", encoding="utf-8") as stream:
            runs = list(csv.DictReader(stream))
        self.assertEqual(signoff["phase_id"], 14)
        self.assertEqual(signoff["artifact_version"], "PERSISTENCE-v1")
        self.assertEqual(signoff["status"], "PASS")
        self.assertEqual(signoff["run_status"], "COMPLETED")
        self.assertEqual(signoff["validation_sample_count"], 2960)
        self.assertFalse(signoff["requires_training"])
        self.assertEqual(signoff["trainable_parameters"], 0)
        self.assertIsNone(signoff["feature_variant_id"])
        self.assertIsNone(signoff["scaler_bundle_id"])
        self.assertIsNone(signoff["seed"])
        self.assertFalse(signoff["test_access_authorized"])
        self.assertFalse(signoff["test_targets_materialized"])
        self.assertEqual(signoff["test_access_policy"], "LOCKED_UNTIL_PHASE_47")
        self.assertEqual(manifest["split_id"], "VALIDATION")
        self.assertEqual(manifest["formula"], "y_hat(t+1) = y(t)")
        self.assertEqual(manifest["source_lag_steps"], 1)
        self.assertEqual(manifest["forecast_horizon_steps"], 1)
        self.assertEqual(manifest["validation_sample_count"], 2960)
        self.assertFalse(manifest["test_targets_materialized"])
        self.assertEqual(metrics["metric_result"]["split_id"], "VALIDATION")
        self.assertEqual(metrics["metric_result"]["n_samples"], 2960)
        self.assertAlmostEqual(metrics["metric_result"]["mae_wh"], 26.16216216216216)
        self.assertAlmostEqual(metrics["metric_result"]["rmse_wh"], 66.42970273458556)
        self.assertAlmostEqual(metrics["metric_result"]["r2"], 0.481325958122121)
        self.assertEqual(len(predictions), 2960)
        self.assertEqual({row["run_id"] for row in predictions}, {signoff["run_id"]})
        persistence_runs = [row for row in runs if row["experiment_family"] == "PERSISTENCE_BASELINE"]
        self.assertEqual(len(persistence_runs), 1)
        self.assertEqual(persistence_runs[0]["run_id"], signoff["run_id"])
        self.assertEqual(persistence_runs[0]["execution_type"], "EVALUATION")
        self.assertEqual(persistence_runs[0]["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
