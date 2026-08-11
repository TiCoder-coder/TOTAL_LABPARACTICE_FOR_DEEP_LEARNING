"""Tests for staged hyperparameter-search configuration and ranking."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


TOTAL_PRACTICE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOTAL_PRACTICE_ROOT))

from practice_1.processing_own_phase.hyperparameter_search import (
    aggregate_confirmation_results,
    build_trial_config,
    export_search_outputs,
    generate_architecture_trials,
    generate_confirmation_trials,
    generate_learning_rate_trials,
    rank_results,
    refinement_learning_rates,
)


BASE_CONFIG = {
    "num_classes": 10,
    "input_dim": 784,
    "weight_decay": 0.0,
    "batch_size": 64,
    "dropout": 0.0,
    "optimizer": "Adam",
    "use_augmentation": False,
}


def make_result(
    config,
    *,
    accuracy,
    loss,
    best_epoch,
    parameters=100,
    seconds=1.0,
):
    return {
        "experiment_id": config["experiment_id"],
        "config": config,
        "best_validation_accuracy": accuracy,
        "best_validation_loss": loss,
        "best_epoch": best_epoch,
        "parameter_count": parameters,
        "total_seconds": seconds,
        "resumed_from_epoch": 0,
        "recovery_checkpoint_path": f"{config['experiment_id']}.pth",
    }


class HyperparameterSearchTests(unittest.TestCase):
    def test_generates_unique_controlled_trial_ids(self) -> None:
        learning_trials = generate_learning_rate_trials(
            BASE_CONFIG,
            (1e-4, 3e-4, 1e-3),
            hidden_dims=(256, 128),
            epochs=20,
            seed=42,
        )
        architecture_trials = generate_architecture_trials(
            BASE_CONFIG,
            ((256,), (256, 128), (256, 128, 64)),
            learning_rate=1e-3,
            epochs=25,
            seed=42,
        )

        self.assertEqual(len(learning_trials), 3)
        self.assertEqual(len(architecture_trials), 3)
        self.assertEqual(
            len({config["experiment_id"] for config in learning_trials}),
            3,
        )
        self.assertEqual(
            {config["hidden_dims"] for config in architecture_trials},
            {(256,), (256, 128), (256, 128, 64)},
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            generate_learning_rate_trials(
                BASE_CONFIG,
                (1e-3, 1e-3),
                hidden_dims=(256, 128),
                epochs=20,
                seed=42,
            )

    def test_refinement_rates_are_bounded_and_ordered(self) -> None:
        rates = refinement_learning_rates(1e-3)

        self.assertEqual(tuple(sorted(rates)), rates)
        self.assertEqual(len(rates), 3)
        self.assertIn(1e-3, rates)
        self.assertTrue(all(1e-5 <= rate <= 1e-2 for rate in rates))

    def test_ranking_uses_accuracy_then_loss_then_parameters(self) -> None:
        configs = generate_learning_rate_trials(
            BASE_CONFIG,
            (1e-4, 3e-4, 1e-3),
            hidden_dims=(256, 128),
            epochs=20,
            seed=42,
        )
        results = [
            make_result(configs[0], accuracy=0.89, loss=0.31, best_epoch=8),
            make_result(configs[1], accuracy=0.90, loss=0.34, best_epoch=9),
            make_result(
                configs[2],
                accuracy=0.90,
                loss=0.30,
                best_epoch=7,
                parameters=200,
            ),
        ]

        ranked = rank_results(results)

        self.assertIs(ranked[0], results[2])
        self.assertIs(ranked[1], results[1])
        self.assertIs(ranked[2], results[0])

    def test_confirmation_aggregation_uses_mean_std_and_median_epoch(self) -> None:
        candidate_config = build_trial_config(
            BASE_CONFIG,
            stage="architecture",
            hidden_dims=(256, 128),
            learning_rate=1e-3,
            epochs=25,
            seed=42,
        )
        confirmation_configs = generate_confirmation_trials(
            BASE_CONFIG,
            (candidate_config,),
            seeds=(42, 123, 2026),
            epochs=40,
        )
        results = [
            make_result(
                config,
                accuracy=accuracy,
                loss=loss,
                best_epoch=best_epoch,
            )
            for config, accuracy, loss, best_epoch in zip(
                confirmation_configs,
                (0.89, 0.90, 0.91),
                (0.32, 0.30, 0.31),
                (12, 18, 15),
            )
        ]

        summaries = aggregate_confirmation_results(results)

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertAlmostEqual(summary["mean_validation_accuracy"], 0.90)
        self.assertGreater(summary["std_validation_accuracy"], 0.0)
        self.assertAlmostEqual(summary["mean_validation_loss"], 0.31)
        self.assertEqual(summary["median_best_epoch"], 15)
        self.assertEqual(summary["trial_count"], 3)

    def test_exports_json_csv_and_best_config_atomically(self) -> None:
        candidate_config = build_trial_config(
            BASE_CONFIG,
            stage="confirmation",
            hidden_dims=(256, 128),
            learning_rate=1e-3,
            epochs=40,
            seed=42,
        )
        result = make_result(
            candidate_config,
            accuracy=0.90,
            loss=0.30,
            best_epoch=15,
        )
        summary = aggregate_confirmation_results((result,))[0]

        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            export_search_outputs(
                (result,),
                (summary,),
                summary,
                output_directory,
            )

            manifest = json.loads(
                (output_directory / "search_summary.json").read_text()
            )
            best = json.loads(
                (output_directory / "best_hyperparameters.json").read_text()
            )
            self.assertEqual(len(manifest["trials"]), 1)
            self.assertEqual(best["median_best_epoch"], 15)
            self.assertIn(
                "validation_accuracy",
                (output_directory / "search_summary.csv").read_text(),
            )
            self.assertFalse(list(output_directory.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
