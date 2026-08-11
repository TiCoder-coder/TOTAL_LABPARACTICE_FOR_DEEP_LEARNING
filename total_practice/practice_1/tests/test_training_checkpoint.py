"""Tests for atomic anti-fail training checkpoints."""

from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch


TOTAL_PRACTICE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOTAL_PRACTICE_ROOT))

from practice_1.processing_own_phase.training_checkpoint import (
    CheckpointCompatibilityError,
    TrainingCheckpointError,
    TrainingCheckpointManager,
    build_checkpoint_signature,
    capture_rng_state,
    fingerprint_indices,
    restore_rng_state,
)


class TrainingCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.checkpoint_path = (
            Path(self.temporary_directory.name) / "experiment.resume.pth"
        )
        self.signature = build_checkpoint_signature({
            "config": {"hidden_dims": (16, 8), "epochs": 2},
            "data": {"sample_count": 12, "indices": "abc"},
        })
        self.manager = TrainingCheckpointManager(
            self.checkpoint_path,
            run_kind="experiment",
            run_id="E0_test",
            signature=self.signature,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_state(
        self,
        *,
        status: str = "in_progress",
        completed_epoch: int = 1,
        total_epochs: int = 2,
    ) -> dict:
        loader_generator = torch.Generator().manual_seed(42)
        return {
            "status": status,
            "completed_epoch": completed_epoch,
            "total_epochs": total_epochs,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "optimizer_state_dict": {
                "state": {},
                "param_groups": [{"lr": 0.001, "params": [0]}],
            },
            "history": [
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "train_accuracy": 0.5,
                }
                for epoch in range(1, completed_epoch + 1)
            ],
            "best_state_dict": {"weight": torch.tensor([1.0])},
            "best_epoch": completed_epoch,
            "best_validation_accuracy": 0.5,
            "best_validation_loss": 1.0,
            "rng_state": capture_rng_state(loader_generator),
            "elapsed_seconds": 1.25,
            "log_directory": "/tmp/test-log",
        }

    def test_signature_and_index_fingerprint_are_stable(self) -> None:
        first = build_checkpoint_signature({
            "b": [1, 2],
            "a": {"dims": (16, 8)},
        })
        second = build_checkpoint_signature({
            "a": {"dims": [16, 8]},
            "b": (1, 2),
        })

        self.assertEqual(first, second)
        self.assertEqual(
            fingerprint_indices([1, 2, 3]),
            fingerprint_indices((1, 2, 3)),
        )
        self.assertNotEqual(
            fingerprint_indices([1, 2, 3]),
            fingerprint_indices([3, 2, 1]),
        )

    def test_atomic_round_trip_uses_cpu_clones(self) -> None:
        state = self.make_state()
        original_weight = state["model_state_dict"]["weight"]
        self.manager.save(state)
        original_weight.add_(10.0)

        loaded = self.manager.load()

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["completed_epoch"], 1)
        torch.testing.assert_close(
            loaded["model_state_dict"]["weight"],
            torch.tensor([1.0]),
        )
        self.assertFalse(list(self.checkpoint_path.parent.glob("*.tmp")))

    def test_disabled_manager_is_a_no_op(self) -> None:
        manager = TrainingCheckpointManager(
            self.checkpoint_path,
            run_kind="experiment",
            run_id="disabled",
            signature=self.signature,
            enabled=False,
        )

        manager.save(self.make_state())

        self.assertFalse(self.checkpoint_path.exists())
        self.assertIsNone(manager.load())

    def test_rejects_incompatible_and_invalid_payloads(self) -> None:
        self.manager.save(self.make_state())
        incompatible_manager = TrainingCheckpointManager(
            self.checkpoint_path,
            run_kind="experiment",
            run_id="different-run",
            signature=self.signature,
        )
        with self.assertRaises(CheckpointCompatibilityError):
            incompatible_manager.load()

        invalid_state = self.make_state(
            status="completed",
            completed_epoch=1,
            total_epochs=2,
        )
        with self.assertRaises(TrainingCheckpointError):
            self.manager.save(invalid_state)

    def test_corrupted_checkpoint_fails_closed(self) -> None:
        self.checkpoint_path.write_bytes(b"not a torch checkpoint")

        with self.assertRaises(TrainingCheckpointError):
            self.manager.load()

    def test_rng_round_trip_restores_all_cpu_sequences(self) -> None:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        loader_generator = torch.Generator().manual_seed(7)
        state = capture_rng_state(loader_generator)

        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(3)
        expected_loader = torch.rand(3, generator=loader_generator)

        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        loader_generator.manual_seed(99)
        restore_rng_state(state, loader_generator)

        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.rand(3, generator=loader_generator),
            expected_loader,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
