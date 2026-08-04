"""Contracts for the controlled Phase 2.5 E3 experiment."""

import torch

from practice_2_2.canonical_e3_practice_2_2 import (
    E2_BASELINE,
    E3_CONFIG,
    EXPERIMENT_ID,
    RUN_ID,
    choose_provisional_winner,
    validate_e3_contract,
)
from practice_2_2.canonical_train_practice_2_2 import EXPERIMENTS
from practice_2_2.model import build_model
from practice_2_2.train import set_frozen_batchnorm_eval


def test_e3_differs_from_canonical_e2_only_by_training_mode():
    assert validate_e3_contract()
    e2 = EXPERIMENTS["E2_partial_finetune"]
    differing = {key for key in set(e2).union(E3_CONFIG) if e2.get(key) != E3_CONFIG.get(key)}
    assert differing == {"training_mode"}
    assert E3_CONFIG["training_mode"] == "last_block_finetune"


def test_e3_resnet18_trainable_depth_and_frozen_batchnorm():
    torch.manual_seed(42)
    model = build_model("resnet18", "last_block_finetune", num_classes=10, dropout=0.2)
    assert not any(p.requires_grad for p in model.network.layer4[0].parameters())
    assert all(p.requires_grad for p in model.network.layer4[1].parameters())
    assert all(p.requires_grad for p in model.network.fc.parameters())
    assert not any(p.requires_grad for p in model.network.layer3.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 4_725_770

    model.train()
    set_frozen_batchnorm_eval(model)
    assert all(not m.training for m in model.network.layer4[0].modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    assert all(m.training for m in model.network.layer4[1].modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))


def test_e3_selection_is_validation_only_and_test_locked():
    assert RUN_ID == "canonical_26dc4625_52aaf974_s42_phase25_e3_layer4_1_v1"
    assert EXPERIMENT_ID == "E3_layer4_1_head"
    assert E3_CONFIG["test_loader_constructed"] is False
    assert E3_CONFIG["test_evaluated"] is False
    better = {**E2_BASELINE, "validation_accuracy": E2_BASELINE["validation_accuracy"] + 0.1}
    assert choose_provisional_winner(better)["winner"] == EXPERIMENT_ID
    worse = {**E2_BASELINE, "validation_accuracy": E2_BASELINE["validation_accuracy"] - 1.0}
    assert choose_provisional_winner(worse)["winner"] == E2_BASELINE["experiment"]
