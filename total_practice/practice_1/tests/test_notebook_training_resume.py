"""Deterministic interruption/resume tests for notebook training functions."""

import json
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Dict, Optional
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


PRACTICE_ROOT = Path(__file__).resolve().parents[1]
TOTAL_PRACTICE_ROOT = PRACTICE_ROOT.parent
sys.path.insert(0, str(TOTAL_PRACTICE_ROOT))

from practice_1.processing_own_phase.training_checkpoint import (
    TrainingCheckpointManager,
    build_checkpoint_signature,
    capture_rng_state,
    restore_rng_state,
)
from practice_1.processing_own_phase.training_monitor import TrainingMonitor


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class InterruptAfterEpoch:
    def __init__(
        self,
        manager: TrainingCheckpointManager,
        epoch: int,
    ) -> None:
        self.manager = manager
        self.epoch = epoch
        self.path = manager.path

    def load(self, map_location: str = "cpu") -> Optional[dict]:
        return self.manager.load(map_location)

    def save(self, state: dict) -> None:
        self.manager.save(state)
        if state["completed_epoch"] == self.epoch:
            raise RuntimeError("simulated interruption")


def assert_nested_equal(
    test_case: unittest.TestCase,
    first,
    second,
) -> None:
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        test_case.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_nested_equal(test_case, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test_case.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            assert_nested_equal(test_case, left, right)
    else:
        test_case.assertEqual(first, second)


class NotebookTrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        notebook = json.loads(
            (PRACTICE_ROOT / "practice_1.ipynb").read_text(encoding="utf-8")
        )
        namespace = {
            "random": random,
            "np": np,
            "torch": torch,
            "nn": nn,
            "optim": optim,
            "time": time,
            "Path": Path,
            "Dict": Dict,
            "Optional": Optional,
            "DataLoader": DataLoader,
            "SummaryWriter": SummaryWriter,
            "TrainingMonitor": TrainingMonitor,
            "TrainingCheckpointManager": TrainingCheckpointManager,
            "capture_rng_state": capture_rng_state,
            "restore_rng_state": restore_rng_state,
            "SEED": 123,
        }
        for cell_index in (62, 63, 64, 65):
            source = "".join(notebook["cells"][cell_index]["source"])
            exec(
                compile(source, f"<cell-{cell_index}>", "exec"),
                namespace,
            )
        namespace["build_model"] = lambda config: TinyModel()
        namespace["count_trainable_parameters"] = lambda model: sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        for cell_index in (67, 76):
            source = "".join(notebook["cells"][cell_index]["source"])
            exec(
                compile(source, f"<cell-{cell_index}>", "exec"),
                namespace,
            )

        cls.run_training = staticmethod(namespace["run_training"])
        cls.train_final_model = staticmethod(namespace["train_final_model"])

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        generator = torch.Generator().manual_seed(2026)
        features = torch.randn(32, 4, generator=generator)
        labels = ((features[:, 0] + features[:, 1]) > 0).long()
        self.training_dataset = TensorDataset(features[:24], labels[:24])
        self.validation_loader = DataLoader(
            TensorDataset(features[24:], labels[24:]),
            batch_size=4,
        )
        self.config = {
            "experiment_id": "tiny",
            "epochs": 4,
            "batch_size": 6,
            "seed": 123,
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
        }
        self.signature = build_checkpoint_signature({
            "config": self.config,
            "data": "tiny-v1",
        })

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def manager(self, name: str, run_kind: str) -> TrainingCheckpointManager:
        return TrainingCheckpointManager(
            self.root / f"{name}.pth",
            run_kind=run_kind,
            run_id="tiny",
            signature=self.signature,
        )

    def test_experiment_resume_matches_uninterrupted_run(self) -> None:
        full_manager = self.manager("full-experiment", "experiment")
        resumed_manager = self.manager("resumed-experiment", "experiment")
        full_result = self.run_training(
            self.config,
            self.training_dataset,
            self.validation_loader,
            torch.device("cpu"),
            verbose=False,
            checkpoint_manager=full_manager,
        )

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_training(
                self.config,
                self.training_dataset,
                self.validation_loader,
                torch.device("cpu"),
                verbose=False,
                checkpoint_manager=InterruptAfterEpoch(resumed_manager, 2),
            )
        interrupted_state = resumed_manager.load()
        self.assertEqual(interrupted_state["status"], "in_progress")
        self.assertEqual(interrupted_state["completed_epoch"], 2)

        resumed_result = self.run_training(
            self.config,
            self.training_dataset,
            self.validation_loader,
            torch.device("cpu"),
            verbose=False,
            checkpoint_manager=resumed_manager,
        )
        self.assertEqual(resumed_result["resumed_from_epoch"], 2)
        self._assert_metric_histories_equal(
            full_result["history"],
            resumed_result["history"],
            include_validation=True,
        )
        assert_nested_equal(
            self,
            full_manager.load()["optimizer_state_dict"],
            resumed_manager.load()["optimizer_state_dict"],
        )
        assert_nested_equal(
            self,
            full_result["model"].state_dict(),
            resumed_result["model"].state_dict(),
        )

    def test_final_resume_matches_uninterrupted_run(self) -> None:
        final_config = {**self.config, "experiment_id": "tiny_final"}
        final_signature = build_checkpoint_signature({
            "config": final_config,
            "data": "tiny-final-v1",
        })
        full_manager = TrainingCheckpointManager(
            self.root / "full-final.pth",
            "final",
            "tiny_final",
            final_signature,
        )
        resumed_manager = TrainingCheckpointManager(
            self.root / "resumed-final.pth",
            "final",
            "tiny_final",
            final_signature,
        )
        full_result = self.train_final_model(
            final_config,
            self.training_dataset,
            torch.device("cpu"),
            verbose=False,
            checkpoint_manager=full_manager,
        )
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.train_final_model(
                final_config,
                self.training_dataset,
                torch.device("cpu"),
                verbose=False,
                checkpoint_manager=InterruptAfterEpoch(resumed_manager, 2),
            )

        resumed_result = self.train_final_model(
            final_config,
            self.training_dataset,
            torch.device("cpu"),
            verbose=False,
            checkpoint_manager=resumed_manager,
        )
        self._assert_metric_histories_equal(
            full_result["history"],
            resumed_result["history"],
            include_validation=False,
        )
        self.assertEqual(resumed_manager.load()["status"], "completed")
        assert_nested_equal(
            self,
            full_result["model"].state_dict(),
            resumed_result["model"].state_dict(),
        )

    def test_monitor_and_tensorboard_continue_without_duplicate_steps(self) -> None:
        manager = self.manager("monitor", "experiment")
        log_directory = self.root / "events"
        first_monitor = TrainingMonitor("first", 4, True)
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            with first_monitor:
                self.run_training(
                    self.config,
                    self.training_dataset,
                    self.validation_loader,
                    torch.device("cpu"),
                    log_directory=log_directory,
                    verbose=False,
                    monitor=first_monitor,
                    checkpoint_manager=InterruptAfterEpoch(manager, 2),
                )

        resumed_monitor = TrainingMonitor("resumed", 4, True)
        with resumed_monitor:
            self.run_training(
                self.config,
                self.training_dataset,
                self.validation_loader,
                torch.device("cpu"),
                log_directory=self.root / "ignored-new-events",
                verbose=False,
                monitor=resumed_monitor,
                checkpoint_manager=manager,
            )
        self.assertEqual(len(resumed_monitor.history), 4)
        self.assertFalse((self.root / "ignored-new-events").exists())

        events = EventAccumulator(str(log_directory))
        events.Reload()
        for tag in (
            "Loss/Train",
            "Loss/Validation",
            "Accuracy/Train",
            "Accuracy/Validation",
            "LearningRate",
        ):
            self.assertEqual(
                [event.step for event in events.Scalars(tag)],
                [1, 2, 3, 4],
            )

    def _assert_metric_histories_equal(
        self,
        full_history: list[dict],
        resumed_history: list[dict],
        include_validation: bool,
    ) -> None:
        metric_keys = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "learning_rate",
        ]
        if include_validation:
            metric_keys.extend(("validation_loss", "validation_accuracy"))
        self.assertEqual(len(full_history), len(resumed_history))
        for full_record, resumed_record in zip(
            full_history,
            resumed_history,
        ):
            for key in metric_keys:
                self.assertEqual(full_record[key], resumed_record[key])


if __name__ == "__main__":
    unittest.main()
