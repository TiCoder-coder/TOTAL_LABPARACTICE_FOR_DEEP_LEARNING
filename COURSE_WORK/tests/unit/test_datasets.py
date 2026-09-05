import unittest

import numpy as np
import pandas as pd
import torch

from course_work.data.datasets import (
    DatasetConfig,
    SequenceWindowDataset,
    TargetAccessMode,
    build_dataloader,
    build_split_generator,
    build_test_evaluation_dataset,
)


class SequenceWindowDatasetTest(unittest.TestCase):
    def build_dataset(self, split_id: str = "TRAIN", size: int = 70) -> SequenceWindowDataset:
        matrix = np.arange((size + 3) * 2, dtype=np.float32).reshape(size + 3, 2)
        records = []
        for index in range(size):
            target = index + 3
            timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=10 * target)
            records.append({
                "window_id": f"W{index}",
                "target_sample_id": f"T{index}",
                "lookback_steps": 3,
                "horizon_steps": 1,
                "timeline_input_start": index,
                "timeline_input_end": index + 2,
                "timeline_target": target,
                "input_start_timestamp": timestamp - pd.Timedelta(minutes=30),
                "input_end_timestamp": timestamp - pd.Timedelta(minutes=10),
                "target_timestamp": timestamp,
                "target_split_id": split_id,
            })
        mode = {
            "TRAIN": TargetAccessMode.TRAIN,
            "VALIDATION": TargetAccessMode.VALIDATION,
            "TEST": TargetAccessMode.TEST_LOCKED,
        }[split_id]
        config = DatasetConfig(
            dataset_config_id=f"DS_{split_id}",
            split_id=split_id,
            variant_id="FS_TEST",
            lookback=3,
            horizon=1,
            target_option="YS0",
            target_access_mode=mode.value,
            feature_count=2,
            feature_fingerprint="feature",
            scaler_bundle_id="scaler",
            scaler_checksum="checksum",
            window_fingerprint="window",
            population_fingerprint="population",
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            dataset_fingerprint="dataset",
        )
        targets = None if split_id == "TEST" else np.arange(size + 3, dtype=np.float64)
        return SequenceWindowDataset(matrix, pd.DataFrame(records), config, targets, audit_mode=True)

    def test_train_item_contract_is_deterministic(self) -> None:
        dataset = self.build_dataset()
        first = dataset[4]
        second = dataset[4]
        self.assertEqual(set(first), {"x", "y_model", "y_raw_wh", "sample_idx"})
        self.assertTrue(torch.equal(first["x"], second["x"]))
        self.assertTrue(torch.equal(first["y_model"], second["y_model"]))
        self.assertEqual(first["x"].shape, (3, 2))
        self.assertEqual(first["y_model"].shape, (1,))
        self.assertEqual(first["x"].dtype, torch.float32)
        self.assertEqual(first["sample_idx"].dtype, torch.int64)

    def test_tensor_mutation_does_not_mutate_dataset_timeline(self) -> None:
        dataset = self.build_dataset()
        expected = dataset[0]["x"].clone()
        changed = dataset[0]["x"]
        changed[0, 0] = -999
        self.assertTrue(torch.equal(dataset[0]["x"], expected))

    def test_test_locked_item_has_no_target(self) -> None:
        dataset = self.build_dataset("TEST")
        self.assertEqual(set(dataset[0]), {"x", "sample_idx"})

    def test_loader_split_policy_and_batch_shapes(self) -> None:
        expected = {
            "TRAIN": (True, {"x", "y_model", "y_raw_wh", "sample_idx"}),
            "VALIDATION": (False, {"x", "y_model", "y_raw_wh", "sample_idx"}),
            "TEST": (False, {"x", "sample_idx"}),
        }
        for split_id, (shuffle, keys) in expected.items():
            loader, config, _ = build_dataloader(self.build_dataset(split_id), 32, 42, 0, "cpu")
            batch = next(iter(loader))
            self.assertEqual(set(batch), keys)
            self.assertEqual(batch["x"].shape, (32, 3, 2))
            self.assertEqual(config.shuffle, shuffle)
            self.assertFalse(config.drop_last)
            self.assertEqual(config.expected_batches, 3)

    def test_b32_and_b64_preserve_every_sample(self) -> None:
        for batch_size in (32, 64):
            dataset = self.build_dataset(size=70)
            loader, config, _ = build_dataloader(dataset, batch_size, 42, 0, "cpu")
            observed = torch.cat([batch["sample_idx"] for batch in loader]).tolist()
            self.assertEqual(len(observed), 70)
            self.assertEqual(len(set(observed)), 70)
            self.assertEqual(config.expected_batches, 3 if batch_size == 32 else 2)

    def test_same_seed_fresh_train_loaders_are_reproducible(self) -> None:
        dataset = self.build_dataset()
        loader_1, _, _ = build_dataloader(dataset, 32, 42, 0, "cpu")
        loader_2, _, _ = build_dataloader(dataset, 32, 42, 0, "cpu")
        order_1 = [int(dataset.sample_indices[index]) for index in loader_1.sampler]
        order_2 = [int(dataset.sample_indices[index]) for index in loader_2.sampler]
        self.assertEqual(order_1, order_2)

    def test_consecutive_train_epochs_change_order(self) -> None:
        dataset = self.build_dataset()
        loader, _, _ = build_dataloader(dataset, 32, 42, 0, "cpu")
        first = list(loader.sampler)
        second = list(loader.sampler)
        self.assertNotEqual(first, second)

    def test_validation_and_test_are_chronological(self) -> None:
        for split_id in ("VALIDATION", "TEST"):
            dataset = self.build_dataset(split_id)
            loader, _, _ = build_dataloader(dataset, 32, 42, 0, "cpu")
            observed = torch.cat([batch["sample_idx"] for batch in loader]).numpy()
            self.assertTrue(np.array_equal(observed, dataset.sample_indices))

    def test_generators_are_split_separated(self) -> None:
        values = {
            split_id: torch.rand(4, generator=build_split_generator(42, split_id))
            for split_id in ("TRAIN", "VALIDATION", "TEST")
        }
        self.assertFalse(torch.equal(values["TRAIN"], values["VALIDATION"]))
        self.assertFalse(torch.equal(values["VALIDATION"], values["TEST"]))

    def test_test_evaluation_factory_rejects_missing_phase_47_authorization(self) -> None:
        with self.assertRaises(PermissionError):
            build_test_evaluation_dataset("DENIED")

    def test_invalid_loader_settings_are_rejected(self) -> None:
        dataset = self.build_dataset()
        with self.assertRaises(ValueError):
            build_dataloader(dataset, 16)
        with self.assertRaises(ValueError):
            build_dataloader(dataset, 32, num_workers=-1)
        with self.assertRaises(ValueError):
            build_dataloader(dataset, 32, device_type="unknown")


if __name__ == "__main__":
    unittest.main()
