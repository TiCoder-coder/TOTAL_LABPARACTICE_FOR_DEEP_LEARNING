"""Fail-closed contracts for the one-time Practice 2.2 Final Test."""

import json

import pytest
import torch
from PIL import Image

from practice_2_2.canonical_train_practice_2_2 import _state_dict_hash
from practice_2_2.data_practice_2_2 import get_practice_2_2_transforms
from practice_2_2.final_test_practice_2_2 import (
    CHECKPOINT_SHA256,
    CLASS_ORDER,
    EXPERIMENT_ID,
    claim_final_test_directory,
    freeze_selection,
)


def test_final_test_directory_claim_is_one_time(tmp_path):
    target = tmp_path / "final_test"
    claim_final_test_directory(target)
    with pytest.raises(FileExistsError):
        claim_final_test_directory(target)


def test_frozen_selection_is_exclusive_and_test_independent(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    selection = freeze_selection(tmp_path, checkpoint)
    assert selection["immutable"] is True
    assert selection["selected_experiment_id"] == EXPERIMENT_ID
    assert selection["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert selection["test_used_for_selection"] is False
    assert selection["winner_change_allowed"] is False
    with pytest.raises(FileExistsError):
        freeze_selection(tmp_path, checkpoint)


def test_final_test_transform_is_deterministic_and_matches_validation():
    _, validation = get_practice_2_2_transforms(224, "base")
    _, final_test = get_practice_2_2_transforms(224, "e4_moderate")
    assert repr(validation) == repr(final_test)
    image = Image.new("RGB", (300, 260), "white")
    assert torch.equal(final_test(image), final_test(image))


def test_class_order_and_inference_do_not_mutate_model():
    assert len(CLASS_ORDER) == 10
    assert len(set(CLASS_ORDER)) == 10
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
    model.eval()
    before = _state_dict_hash(model)
    with torch.inference_mode():
        model(torch.ones(3, 2))
    assert _state_dict_hash(model) == before
