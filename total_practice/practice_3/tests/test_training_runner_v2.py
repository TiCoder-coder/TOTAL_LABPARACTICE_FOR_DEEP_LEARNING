"""Safe Part 3.2 runner tests; this module never calls run_experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from processing_own_phase.experiment_protocol_v2 import (
    LEARNING_RATES,
    PROTOCOL_VERSION,
    build_run_config,
    atomic_write_json,
)
from processing_own_phase.experiment_registry_v2 import (
    _registry_record,
    persist_run_transition,
)
from processing_own_phase.experiment_runner_v2 import (
    build_parser,
    checkpoint_fingerprint,
    collect_training_history,
    preflight_run,
    select_best_epoch,
)


class RunnerSafetyTests(unittest.TestCase):
    def test_cli_requires_one_known_run_id(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(["--run-id", "p3v2_lr_2e-5"])
        self.assertEqual(parsed.run_id, "p3v2_lr_2e-5")
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--run-id", "unknown"])

    def test_real_completed_run_cannot_be_started_again(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must be PLANNED, got COMPLETED"):
            preflight_run("p3v2_lr_2e-5")

    def test_history_selection_and_checkpoint_fingerprint(self) -> None:
        logs = [
            {"loss": 0.6, "learning_rate": 2e-5, "epoch": 1.0, "step": 10},
            {"eval_loss": 0.4, "eval_accuracy": 0.8, "eval_precision": 0.8,
             "eval_recall": 0.8, "eval_f1": 0.8, "epoch": 1.0, "step": 10},
            {"loss": 0.5, "learning_rate": 1e-5, "epoch": 2.0, "step": 20},
            {"eval_loss": 0.4000005, "eval_accuracy": 0.82, "eval_precision": 0.81,
             "eval_recall": 0.83, "eval_f1": 0.82, "epoch": 2.0, "step": 20},
        ]
        history = collect_training_history(logs)
        self.assertEqual(select_best_epoch(history)["epoch"], 2.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "model.safetensors").write_bytes(b"synthetic-only")
            result = checkpoint_fingerprint(checkpoint)
            self.assertEqual(result["checkpoint_weight_file"], "model.safetensors")
            self.assertEqual(len(result["checkpoint_hash"]), 64)

    def test_registry_persistence_uses_only_temporary_state(self) -> None:
        configs = [build_run_config(rate, "dataset", "split") for rate in LEARNING_RATES]
        registry = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "NOT_STARTED",
            "runs": [_registry_record(config) for config in configs],
            "real_experiments_executed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            atomic_write_json(path, registry)
            updated = persist_run_transition(
                path, configs, configs[0]["run_id"], "RUNNING",
                config_hash=configs[0]["config_hash"], expected_status="PLANNED",
            )
            self.assertEqual(updated["runs"][0]["status"], "RUNNING")
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["runs"][0]["status"], "RUNNING")
            with self.assertRaises(RuntimeError):
                persist_run_transition(
                    path, configs, configs[0]["run_id"], "RUNNING",
                    config_hash=configs[0]["config_hash"], expected_status="PLANNED",
                )


if __name__ == "__main__":
    unittest.main()
