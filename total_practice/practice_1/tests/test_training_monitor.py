"""Tests for the epoch-level training monitor."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOTAL_PRACTICE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOTAL_PRACTICE_ROOT))

from practice_1.processing_own_phase.training_monitor import TrainingMonitor


class TrainingMonitorTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    @staticmethod
    def validation_record(epoch: int) -> dict[str, float]:
        return {
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "train_accuracy": 0.5 + epoch * 0.1,
            "validation_loss": 1.2 / epoch,
            "validation_accuracy": 0.4 + epoch * 0.1,
            "learning_rate": 0.001,
            "elapsed_seconds": float(epoch),
        }

    def test_validation_history_and_figure(self) -> None:
        monitor = TrainingMonitor("experiment", 3, include_validation=True)

        for epoch in range(1, 4):
            monitor.update(
                self.validation_record(epoch),
                best_epoch=epoch,
            )

        self.assertEqual(len(monitor.history), 3)
        self.assertEqual(len(monitor.axes), 2)
        self.assertIsNotNone(monitor.figure)
        self.assertEqual(
            [line.get_label() for line in monitor.axes[0].get_lines()],
            ["Train", "Validation", "Best epoch: 3"],
        )
        self.assertEqual(
            list(monitor.axes[0].get_lines()[0].get_xdata()),
            [1, 2, 3],
        )

        copied_history = monitor.history
        copied_history[0]["train_loss"] = 99.0
        self.assertNotEqual(monitor.history[0]["train_loss"], 99.0)

        figure_number = monitor.figure.number
        monitor.close()
        self.assertTrue(monitor.closed)
        self.assertFalse(plt.fignum_exists(figure_number))

    def test_train_only_mode(self) -> None:
        monitor = TrainingMonitor("final", 2, include_validation=False)
        monitor.update({
            "epoch": 1,
            "train_loss": 0.8,
            "train_accuracy": 0.7,
        })

        self.assertEqual(len(monitor.history), 1)
        self.assertEqual(
            [line.get_label() for line in monitor.axes[0].get_lines()],
            ["Train"],
        )
        monitor.close()

    def test_restore_history_renders_once_and_continues(self) -> None:
        monitor = TrainingMonitor("resumed", 3, include_validation=True)
        render_count = 0
        original_render = monitor._render

        def counted_render() -> None:
            nonlocal render_count
            render_count += 1
            original_render()

        monitor._render = counted_render
        monitor.restore(
            [self.validation_record(1), self.validation_record(2)],
            best_epoch=2,
        )

        self.assertEqual(render_count, 1)
        self.assertEqual(len(monitor.history), 2)
        monitor.update(self.validation_record(3), best_epoch=3)
        self.assertEqual(render_count, 2)
        self.assertEqual(len(monitor.history), 3)
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            monitor.restore([self.validation_record(1)], best_epoch=1)
        monitor.close()

    def test_long_history_uses_adaptive_ticks_and_saves_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "plots" / "trial.png"
            monitor = TrainingMonitor(
                "long-run",
                40,
                include_validation=False,
                save_path=save_path,
            )
            for epoch in range(1, 41):
                monitor.update({
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "train_accuracy": min(0.5 + epoch * 0.01, 0.99),
                })

            monitor.figure.canvas.draw()
            visible_ticks = [
                tick
                for tick in monitor.axes[0].get_xticks()
                if 1 <= tick <= 40
            ]
            self.assertLessEqual(len(visible_ticks), 11)
            monitor.close()

            self.assertTrue(save_path.is_file())
            self.assertGreater(save_path.stat().st_size, 0)
            self.assertFalse(list(save_path.parent.glob("*.tmp")))

    def test_disabled_monitor_is_no_op(self) -> None:
        monitor = TrainingMonitor(
            "disabled",
            2,
            include_validation=True,
            enabled=False,
        )
        monitor.update({"invalid": math.nan})

        self.assertEqual(monitor.history, ())
        self.assertIsNone(monitor.figure)
        monitor.close()
        self.assertTrue(monitor.closed)

    def test_rejects_invalid_metric_records(self) -> None:
        with self.subTest("missing validation metrics"):
            monitor = TrainingMonitor("missing", 1, True)
            with self.assertRaises(KeyError):
                monitor.update({
                    "epoch": 1,
                    "train_loss": 1.0,
                    "train_accuracy": 0.5,
                })

        with self.subTest("non-finite loss"):
            monitor = TrainingMonitor("non-finite", 1, False)
            with self.assertRaises(FloatingPointError):
                monitor.update({
                    "epoch": 1,
                    "train_loss": math.nan,
                    "train_accuracy": 0.5,
                })

        with self.subTest("accuracy outside range"):
            monitor = TrainingMonitor("range", 1, False)
            with self.assertRaises(ValueError):
                monitor.update({
                    "epoch": 1,
                    "train_loss": 1.0,
                    "train_accuracy": 1.1,
                })

        with self.subTest("non-sequential epoch"):
            monitor = TrainingMonitor("sequence", 3, False)
            monitor.update({
                "epoch": 1,
                "train_loss": 1.0,
                "train_accuracy": 0.5,
            })
            with self.assertRaises(ValueError):
                monitor.update({
                    "epoch": 3,
                    "train_loss": 0.5,
                    "train_accuracy": 0.6,
                })
            monitor.close()

    def test_context_manager_closes_after_exception(self) -> None:
        monitor = TrainingMonitor("context", 1, False)

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with monitor:
                monitor.update({
                    "epoch": 1,
                    "train_loss": 1.0,
                    "train_accuracy": 0.5,
                })
                raise RuntimeError("training failed")

        self.assertTrue(monitor.closed)
        self.assertIsNotNone(monitor.figure)
        self.assertFalse(plt.fignum_exists(monitor.figure.number))


if __name__ == "__main__":
    unittest.main()
