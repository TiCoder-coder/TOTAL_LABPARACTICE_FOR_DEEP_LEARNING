import unittest
from pathlib import Path

import pandas as pd

from course_work.data.temporal import load_validated_temporal_view
from course_work.data.splitting import (
    SPLIT_IDS,
    TEST_DISTRIBUTION_STATUS,
    build_boundary_neighborhood,
    build_chronological_membership,
    build_split_audit,
    build_train_validation_distribution_summary,
    compute_split_boundaries,
    global_split_fingerprint,
    split_fingerprint,
    validate_split_ratios,
)


class ChronologicalSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.temporal_view = load_validated_temporal_view(cls.root)

    def test_canonical_floor_boundaries_and_counts(self) -> None:
        boundaries = compute_split_boundaries(len(self.temporal_view))
        self.assertEqual(boundaries, {
            "TRAIN": (0, 13814),
            "VALIDATION": (13814, 16774),
            "TEST": (16774, 19735),
        })
        membership = build_chronological_membership(self.temporal_view)
        self.assertEqual(membership["split_id"].value_counts().to_dict(), {
            "TRAIN": 13814,
            "VALIDATION": 2960,
            "TEST": 2961,
        })
        for split_id in SPLIT_IDS:
            subset = membership[membership["split_id"].eq(split_id)]
            self.assertEqual(subset["split_position"].tolist(), list(range(len(subset))))

    def test_membership_preserves_order_and_is_deterministic(self) -> None:
        first = build_chronological_membership(self.temporal_view)
        second = build_chronological_membership(self.temporal_view)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["raw_row_index"].equals(self.temporal_view["raw_row_index"]))
        self.assertTrue(first["timestamp"].equals(self.temporal_view["timestamp_parsed"]))
        self.assertEqual(global_split_fingerprint(first), global_split_fingerprint(second))
        self.assertEqual(len({split_fingerprint(first, split_id) for split_id in SPLIT_IDS}), 3)

    def test_chronology_and_structural_audits_pass(self) -> None:
        membership = build_chronological_membership(self.temporal_view)
        audit = build_split_audit(self.temporal_view, membership, )
        self.assertEqual({row["status"] for row in audit}, {"PASS"})
        train = membership[membership["split_id"].eq("TRAIN")]
        validation = membership[membership["split_id"].eq("VALIDATION")]
        test = membership[membership["split_id"].eq("TEST")]
        self.assertLess(train["timestamp"].iloc[-1], validation["timestamp"].iloc[0])
        self.assertLess(validation["timestamp"].iloc[-1], test["timestamp"].iloc[0])
        self.assertEqual(TEST_DISTRIBUTION_STATUS, "LOCKED_UNTIL_PHASE_47")

    def test_boundary_neighborhood_contains_only_structural_metadata(self) -> None:
        membership = build_chronological_membership(self.temporal_view)
        neighborhood = pd.DataFrame(build_boundary_neighborhood(membership))
        self.assertEqual(len(neighborhood), 20)
        self.assertEqual(set(neighborhood["boundary_id"]), {"TRAIN_TO_VALIDATION", "VALIDATION_TO_TEST"})
        self.assertEqual(set(neighborhood["offset_from_new_split_start"]), set(range(-5, 5)))
        self.assertFalse(set(neighborhood.columns).intersection(self.temporal_view.columns.difference(["raw_row_index", "timestamp_parsed", "continuity_segment_id"])))

    def test_distribution_summary_excludes_test(self) -> None:
        membership = build_chronological_membership(self.temporal_view)
        summary = pd.DataFrame(build_train_validation_distribution_summary(self.temporal_view, membership))
        self.assertEqual(set(summary["split"]), {"TRAIN", "VALIDATION"})
        self.assertNotIn("TEST", summary["split"].tolist())
        self.assertEqual(len(summary), 12)

    def test_invalid_ratios_and_timestamps_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_split_ratios(0.7, 0.2, 0.2)
        with self.assertRaisesRegex(ValueError, "empty split"):
            compute_split_boundaries(3, 0.98, 0.01, 0.01)
        candidate = self.temporal_view.iloc[:10].copy(deep=True)
        unsorted = candidate.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "chronologically sorted"):
            build_chronological_membership(unsorted)
        duplicated = pd.concat([candidate, candidate.iloc[[0]]], ignore_index=True).sort_values("timestamp_parsed", kind="stable").reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_chronological_membership(duplicated)
        null_timestamp = candidate.copy(deep=True)
        null_timestamp.loc[0, "timestamp_parsed"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "non-null datetime"):
            build_chronological_membership(null_timestamp)


if __name__ == "__main__":
    unittest.main()
