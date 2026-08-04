"""Contracts for controlled Phase 2.6 E4 augmentation."""

import torch
from PIL import Image
from torchvision import transforms

from practice_2_2.canonical_e4_practice_2_2 import (
    E2_BASELINE,
    E4_CONFIG,
    EXPERIMENT_ID,
    RUN_ID,
    choose_provisional_winner,
    validate_e4_contract,
)
from practice_2_2.canonical_train_practice_2_2 import EXPERIMENTS
from practice_2_2.data_practice_2_2 import get_practice_2_2_transforms
from practice_2_2.model import build_model


def test_e4_differs_from_e2_only_by_augmentation():
    assert validate_e4_contract()
    e2 = EXPERIMENTS["E2_partial_finetune"]
    differing = {key for key in set(e2).union(E4_CONFIG) if e2.get(key) != E4_CONFIG.get(key)}
    assert differing == {"augment_strength"}
    assert E4_CONFIG["training_mode"] == "partial_finetune"


def test_e4_exact_augmentation_and_unchanged_validation():
    e2_train, e2_val = get_practice_2_2_transforms(224, "base")
    e4_train, e4_val = get_practice_2_2_transforms(224, "e4_moderate")
    assert repr(e2_val) == repr(e4_val)
    kinds = [type(item) for item in e4_train.transforms]
    assert kinds == [
        transforms.RandomResizedCrop,
        transforms.RandomHorizontalFlip,
        transforms.ColorJitter,
        transforms.RandomRotation,
        transforms.RandomAffine,
        transforms.ToTensor,
        transforms.RandomErasing,
        transforms.Normalize,
    ]
    assert e4_train.transforms[0].scale == (0.7, 1.0)
    assert e4_train.transforms[1].p == 0.5
    assert e4_train.transforms[3].degrees == [-8.0, 8.0]
    assert e4_train.transforms[4].translate == (0.05, 0.05)
    assert e4_train.transforms[4].scale == (0.95, 1.05)
    assert e4_train.transforms[6].p == 0.2

    image = Image.new("RGB", (300, 260), "white")
    assert torch.equal(e2_val(image), e2_val(image))
    assert torch.equal(e4_val(image), e4_val(image))


def test_e4_uses_full_layer4_and_same_parameter_count_as_e2():
    torch.manual_seed(42)
    model = build_model("resnet18", "partial_finetune", num_classes=10, dropout=0.2)
    assert all(p.requires_grad for p in model.network.layer4.parameters())
    assert all(p.requires_grad for p in model.network.fc.parameters())
    assert not any(p.requires_grad for p in model.network.layer3.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 8_398_858


def test_e4_validation_only_selection_and_test_lock():
    assert RUN_ID == "canonical_26dc4625_52aaf974_s42_phase26_e4_augmentation_v1"
    assert EXPERIMENT_ID == "E4_moderate_augmentation"
    assert E4_CONFIG["test_loader_constructed"] is False
    assert E4_CONFIG["test_evaluated"] is False
    better = {**E2_BASELINE, "validation_accuracy": E2_BASELINE["validation_accuracy"] + 0.1}
    assert choose_provisional_winner(better)["winner"] == EXPERIMENT_ID
    worse = {**E2_BASELINE, "validation_accuracy": E2_BASELINE["validation_accuracy"] - 1.0}
    assert choose_provisional_winner(worse)["winner"] == E2_BASELINE["experiment"]
