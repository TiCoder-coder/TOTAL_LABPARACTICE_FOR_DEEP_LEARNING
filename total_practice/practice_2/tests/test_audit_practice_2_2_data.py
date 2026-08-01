"""Unit tests for the Practice 2.2 data audit helpers."""

import pandas as pd

from processing_own_phase.audit_practice_2_2_data import (
    _perceptual_cross_boundary_summary,
    assign_group_splits,
    is_generated_augmentation,
    source_stem,
)


def test_generated_augmentation_name_parsing():
    assert is_generated_augmentation("serum_00001_aug007.jpg")
    assert not is_generated_augmentation("serum_00001.jpg")
    assert source_stem("serum_00001_aug007.jpg") == "serum_00001"


def test_original_only_policy_excludes_offline_generated_files():
    rows = []
    for label in range(2):
        for index in range(10):
            group = f"{label:02d}/sample_{index:02d}"
            rows.append(
                {
                    "path": f"sample_{index:02d}.jpg",
                    "class_name": str(label),
                    "label": label,
                    "group": group,
                    "is_augmented_file": False,
                }
            )
            if index == 0:
                rows.append(
                    {
                        "path": f"sample_{index:02d}_aug000.jpg",
                        "class_name": str(label),
                        "label": label,
                        "group": group,
                        "is_augmented_file": True,
                    }
                )
    result = assign_group_splits(pd.DataFrame(rows), seed=42)
    assert not result.loc[
        result["used_original_only_policy"], "is_augmented_file"
    ].any()
    assert result.groupby("group")["split"].nunique().max() == 1


def test_perceptual_audit_flags_cross_split_candidate_but_skips_same_group(monkeypatch):
    records = pd.DataFrame(
        [
            {
                "path": "train.jpg",
                "class_name": "serum",
                "group": "06/source_a",
                "split": "Train",
                "dhash": 0b0000,
            },
            {
                "path": "validation.jpg",
                "class_name": "serum",
                "group": "06/source_b",
                "split": "Validation",
                "dhash": 0b0001,
            },
            {
                "path": "train_aug.jpg",
                "class_name": "serum",
                "group": "06/source_a",
                "split": "Train",
                "dhash": 0b0011,
            },
        ]
    )
    monkeypatch.setattr(
        "processing_own_phase.audit_practice_2_2_data._pixel_mae",
        lambda left, right: 0.0,
    )
    result = _perceptual_cross_boundary_summary(records, max_distance=1)
    assert result["cross_split_candidate_pairs"] == 2
    assert result["cross_split_near_identical_pairs"] == 2
    assert result["cross_class_candidate_pairs"] == 0
