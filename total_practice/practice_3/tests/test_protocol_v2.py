"""Synthetic/static tests for Practice 3 v2 protocol infrastructure."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from processing_own_phase.dataset_protocol_v2 import create_split_manifest
from processing_own_phase.experiment_protocol_v2 import (
    LEARNING_RATES,
    PROTOCOL_VERSION,
    build_run_config,
    optimizer_compatibility_preflight,
    rank_experiments,
    sha256_payload,
    validate_run_configs,
)
from processing_own_phase.experiment_registry_v2 import (
    _registry_record,
    completed_run_action,
    transition_run,
    validate_registry,
)
from processing_own_phase.experiment_runner_v2 import (
    build_early_stopping_callback,
    build_training_arguments,
    prepare_run_context,
)
from processing_own_phase.holdout_guard_v2 import (
    build_winner_manifest,
    claim_holdout_attempt,
    initial_holdout_state,
    guarded_holdout_request,
    unlock_state_after_valid_winner,
)
from processing_own_phase.experiment_protocol_v2 import atomic_write_json


def synthetic_records() -> list[dict]:
    records = []
    for label in (0, 1):
        for index in range(6):
            records.append({
                "source_id": f"train:{label}-{index}",
                "original_split": "train",
                "original_index": label * 10 + index,
                "label": label,
                "text": f"unique-{label}-{index}",
            })
    return records


def synthetic_split() -> dict:
    return create_split_manifest(
        synthetic_records(),
        {"train": "synthetic", "validation": "synthetic"},
        split_counts={"train": 4, "validation": 4, "holdout": 4},
        class_counts={
            "train": {"0": 2, "1": 2},
            "validation": {"0": 2, "1": 2},
            "holdout": {"0": 2, "1": 2},
        },
    )


def configs() -> list[dict]:
    return [build_run_config(rate, "dataset-hash", "split-hash") for rate in LEARNING_RATES]


def planned_registry(run_configs: list[dict]) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "runs": [_registry_record(config) for config in run_configs],
    }


def completed_registry(run_configs: list[dict]) -> dict:
    registry = planned_registry(run_configs)
    metrics = (
        (0.40, 0.80, 0.81, 2),
        (0.30, 0.82, 0.83, 3),
        (0.35, 0.84, 0.85, 1),
    )
    for record, values in zip(registry["runs"], metrics):
        loss, f1, accuracy, epoch = values
        record.update({
            "status": "COMPLETED",
            "stopped_epoch": epoch + 2,
            "best_epoch": epoch,
            "best_val_loss": loss,
            "best_val_accuracy": accuracy,
            "best_val_precision": 0.82,
            "best_val_recall": 0.83,
            "best_val_f1": f1,
            "best_checkpoint": f"checkpoint-{epoch}",
            "runtime_seconds": 1.0,
        })
    return registry


class DatasetProtocolTests(unittest.TestCase):
    def test_deterministic_balanced_split_and_no_overlap(self) -> None:
        first = synthetic_split()
        second = synthetic_split()
        self.assertEqual(first["split_manifest_hash"], second["split_manifest_hash"])
        self.assertEqual(first["split_counts"], {"train": 4, "validation": 4, "holdout": 4})
        self.assertTrue(all(first["validation_checks"].values()))
        self.assertNotIn("text", json.dumps(first["source_ids"]))

    def test_conflicting_duplicate_labels_rejected(self) -> None:
        records = synthetic_records()
        records[1]["text"] = records[0]["text"]
        records[1]["label"] = 1
        with self.assertRaises(ValueError):
            create_split_manifest(
                records,
                {},
                split_counts={"train": 4, "validation": 4, "holdout": 4},
                class_counts={
                    "train": {"0": 2, "1": 2},
                    "validation": {"0": 2, "1": 2},
                    "holdout": {"0": 2, "1": 2},
                },
            )


class ConfigRegistryTests(unittest.TestCase):
    def test_three_configs_only_lr_differs_and_hashes_stable(self) -> None:
        first = configs()
        second = configs()
        validate_run_configs(first)
        self.assertEqual(
            [item["config_hash"] for item in first],
            [item["config_hash"] for item in second],
        )

    def test_invalid_lr_and_duplicate_registry_rejected(self) -> None:
        run_configs = configs()
        registry = planned_registry(run_configs)
        registry["runs"][0]["learning_rate"] = 9e-5
        with self.assertRaises(ValueError):
            validate_registry(registry, run_configs)
        duplicate = planned_registry(run_configs)
        duplicate["runs"][1]["run_id"] = duplicate["runs"][0]["run_id"]
        with self.assertRaises(ValueError):
            validate_registry(duplicate, run_configs)

    def test_lifecycle_and_completed_cache_guard(self) -> None:
        run_configs = configs()
        registry = planned_registry(run_configs)
        running = transition_run(
            registry,
            run_configs[0]["run_id"],
            "RUNNING",
            config_hash=run_configs[0]["config_hash"],
        )
        with self.assertRaises(ValueError):
            transition_run(
                registry,
                run_configs[0]["run_id"],
                "COMPLETED",
                config_hash=run_configs[0]["config_hash"],
            )
        self.assertEqual(
            completed_run_action(
                {"status": "COMPLETED", "config_hash": "abc"}, "abc", True
            ),
            "LOAD",
        )
        self.assertEqual(running["runs"][0]["status"], "RUNNING")


class RankingGuardTests(unittest.TestCase):
    def _records(self) -> list[dict]:
        return [
            {"run_id": "p3v2_lr_2e-5", "status": "COMPLETED", "best_val_loss": 0.4, "best_val_f1": 0.8, "best_val_accuracy": 0.8, "best_epoch": 3},
            {"run_id": "p3v2_lr_3e-5", "status": "COMPLETED", "best_val_loss": 0.3, "best_val_f1": 0.7, "best_val_accuracy": 0.7, "best_epoch": 3},
            {"run_id": "p3v2_lr_5e-5", "status": "COMPLETED", "best_val_loss": 0.5, "best_val_f1": 0.9, "best_val_accuracy": 0.9, "best_epoch": 1},
        ]

    def test_loss_f1_accuracy_and_epoch_ranking(self) -> None:
        records = self._records()
        self.assertEqual(rank_experiments(records)[0]["run_id"], "p3v2_lr_3e-5")
        records[0].update(best_val_loss=0.3 + 5e-7, best_val_f1=0.9)
        self.assertEqual(rank_experiments(records)[0]["run_id"], "p3v2_lr_2e-5")
        records[0].update(best_val_f1=0.7 + 5e-7, best_val_accuracy=0.9)
        self.assertEqual(rank_experiments(records)[0]["run_id"], "p3v2_lr_2e-5")
        records[0].update(best_val_accuracy=0.7 + 5e-7, best_epoch=2)
        self.assertEqual(rank_experiments(records)[0]["run_id"], "p3v2_lr_2e-5")

    def test_holdout_fields_rejected(self) -> None:
        records = self._records()
        records[0]["holdout_accuracy"] = 1.0
        with self.assertRaises(ValueError):
            rank_experiments(records)

    def test_synthetic_winner_and_one_time_claim(self) -> None:
        run_configs = configs()
        registry = completed_registry(run_configs)
        config_map = {item["run_id"]: item for item in run_configs}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = {}
            for run_id in config_map:
                path = root / f"{run_id}.bin"
                path.write_bytes(run_id.encode("utf-8"))
                checkpoints[run_id] = path
            winner = build_winner_manifest(
                registry,
                config_map,
                checkpoints,
                "dataset-hash",
                "split-hash",
                locked_at="2026-08-15T00:00:00+00:00",
            )
            state = unlock_state_after_valid_winner(
                initial_holdout_state(),
                winner,
                registry,
                config_map,
                "dataset-hash",
                "split-hash",
            )
            state_path = root / "holdout_state.json"
            atomic_write_json(state_path, state)
            claimed = claim_holdout_attempt(state_path, winner["winner_manifest_hash"])
            self.assertEqual(claimed["holdout_evaluation_count"], 1)
            with self.assertRaises(PermissionError):
                claim_holdout_attempt(state_path, winner["winner_manifest_hash"])

    def test_winner_rejects_noncompleted_and_wrong_hash(self) -> None:
        run_configs = configs()
        registry = planned_registry(run_configs)
        with self.assertRaises(ValueError):
            build_winner_manifest(registry, {}, {}, "dataset-hash", "split-hash")

    def test_real_style_sealed_guard_rejects_before_provider(self) -> None:
        calls = {"count": 0}
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            atomic_write_json(state_path, initial_holdout_state())

            def provider() -> None:
                calls["count"] += 1

            with self.assertRaises(PermissionError):
                guarded_holdout_request(state_path, "missing-winner", provider)
            self.assertEqual(calls["count"], 0)


class EnvironmentTests(unittest.TestCase):
    def test_optimizer_preflight_without_step(self) -> None:
        result = optimizer_compatibility_preflight()
        self.assertEqual(result["status"], "SUPPORTED")
        self.assertFalse(result["backward_called"])
        self.assertFalse(result["optimizer_step_called"])

    def test_training_arguments_and_early_stopping_contract(self) -> None:
        config = configs()[0]
        context = prepare_run_context(config, Path.cwd())
        arguments = build_training_arguments(config, context)
        callback = build_early_stopping_callback(config)
        self.assertEqual(arguments.num_train_epochs, 10)
        self.assertEqual(str(arguments.eval_strategy.value), "epoch")
        self.assertEqual(str(arguments.save_strategy.value), "epoch")
        self.assertTrue(arguments.load_best_model_at_end)
        self.assertEqual(arguments.metric_for_best_model, "eval_loss")
        self.assertFalse(arguments.greater_is_better)
        self.assertEqual(callback.early_stopping_patience, 2)
        self.assertEqual(callback.early_stopping_threshold, 1e-6)


class ReadinessManifestTests(unittest.TestCase):
    def test_real_readiness_manifest_is_self_consistent(self) -> None:
        path = Path("docs/result/practice_3_v2/pretraining_readiness_manifest.json")
        self.assertTrue(path.is_file())
        manifest = json.loads(path.read_text(encoding="utf-8"))
        supplied = manifest.pop("readiness_manifest_hash")
        self.assertEqual(supplied, sha256_payload(manifest))
        self.assertTrue(manifest["ready_for_training"])
        self.assertFalse(manifest["official_test_loaded"])
        self.assertFalse(manifest["holdout_materialized"])
        self.assertEqual(manifest["holdout_evaluation_count"], 0)
        self.assertFalse(manifest["training_performed"])


if __name__ == "__main__":
    unittest.main()
