"""Unit tests for the regularization primitives added for Practice 2.2."""

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from processing_own_phase.regularization_practice_2_2 import (
    ModelEMA,
    mixup_cutmix_batch,
    soft_cross_entropy,
)
from processing_own_phase.train import WarmupCosineLR


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_mixup_cutmix_identity_when_alphas_zero():
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 1])
    out_images, soft = mixup_cutmix_batch(
        images, labels, mixup_alpha=0.0, cutmix_alpha=0.0
    )
    assert torch.equal(out_images, images)
    assert soft.shape == (4, 3)
    expected = torch.nn.functional.one_hot(labels, num_classes=3).float()
    assert torch.allclose(soft, expected)


def test_mixup_preserves_label_sum_to_one():
    images = torch.randn(8, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 1, 2])
    g = torch.Generator().manual_seed(0)
    out_images, soft = mixup_cutmix_batch(
        images, labels, mixup_alpha=0.4, cutmix_alpha=0.0, generator=g
    )
    assert out_images.shape == images.shape
    sums = soft.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_cutmix_replaces_a_subregion():
    # Use a non-trivial input so the patched pixels are guaranteed to differ
    # from the source image even after a chance swap of identical zeros.
    images = torch.arange(4 * 3 * 16 * 16, dtype=torch.float32).reshape(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 1])
    g = torch.Generator().manual_seed(42)
    out_images, soft = mixup_cutmix_batch(
        images,
        labels,
        mixup_alpha=0.0,
        cutmix_alpha=1.0,
        cutmix_prob=1.0,
        generator=g,
    )
    # Pixels in the patch should differ from the original image.
    assert (out_images != images).any()
    assert torch.allclose(soft.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_mixup_handles_batch_size_one():
    images = torch.randn(1, 3, 8, 8)
    labels = torch.tensor([0])
    out_images, soft = mixup_cutmix_batch(images, labels, mixup_alpha=0.2)
    assert out_images.shape == images.shape
    assert torch.equal(out_images, images)
    assert soft.shape == (1, 1)


def test_soft_cross_entropy_matches_ce_for_one_hot():
    # Without label smoothing the two formulations are identical because
    # soft_cross_entropy reduces to sum over one-hot, i.e. log p_true.
    logits = torch.randn(4, 5)
    labels = torch.tensor([0, 2, 4, 1])
    one_hot = torch.nn.functional.one_hot(labels, num_classes=5).float()
    loss_soft = soft_cross_entropy(logits, one_hot, label_smoothing=0.0)
    loss_hard = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.0)
    assert torch.allclose(loss_soft, loss_hard, atol=1e-5)


def test_soft_cross_entropy_preserves_mixing_target():
    # When two one-hot labels are blended with mixup-style factor ``lam``,
    # the soft loss should equal ``lam * CE(a) + (1-lam) * CE(b)`` for the
    # same logits.
    logits = torch.randn(2, 5)
    target_a = torch.nn.functional.one_hot(torch.tensor([0, 1]), num_classes=5).float()
    target_b = torch.nn.functional.one_hot(torch.tensor([3, 4]), num_classes=5).float()
    lam = 0.4
    blended = lam * target_a + (1 - lam) * target_b
    loss_soft = soft_cross_entropy(logits, blended, label_smoothing=0.0)
    expected = lam * torch.nn.functional.cross_entropy(
        logits, target_a.argmax(dim=-1)
    ) + (1 - lam) * torch.nn.functional.cross_entropy(
        logits, target_b.argmax(dim=-1)
    )
    assert torch.allclose(loss_soft, expected, atol=1e-5)


def test_model_ema_converges_to_trained_weights():
    model = _TinyModel()
    target = _TinyModel()
    target.load_state_dict(model.state_dict())
    ema = ModelEMA(model, decay=0.5)

    # Move target weights a small step.
    with torch.no_grad():
        for p, tp in zip(model.parameters(), target.parameters()):
            p.add_(torch.randn_like(p) * 0.1)
            tp.copy_(p)

    # Many updates should bring EMA close to the live model.
    for _ in range(50):
        ema.update(model)

    for ema_p, target_p in zip(ema.module().parameters(), target.parameters()):
        assert torch.allclose(ema_p, target_p, atol=1e-3)


def test_model_ema_state_round_trip(tmp_path: Path):
    model = _TinyModel()
    ema = ModelEMA(model, decay=0.9)
    state = ema.state_dict()
    path = tmp_path / "ema.pt"
    torch.save(state, path)
    restored = ModelEMA(model, decay=0.1)
    restored.load_state_dict(torch.load(path))
    for p1, p2 in zip(ema.module().parameters(), restored.module().parameters()):
        assert torch.allclose(p1, p2)


def test_warmup_cosine_schedule_monotone_after_warmup():
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = WarmupCosineLR(
        optimizer, warmup_epochs=2, max_epochs=10, min_lr_ratio=0.1
    )
    # Run max_epochs + 1 step cycles so we reach the cosine tail end.
    # (scheduler.step() advances last_epoch first; the final lr at
    # last_epoch == max_epochs is the cosine terminal value.)
    lrs = []
    for _ in range(11):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    # Warmup phase: lrs increase.
    assert lrs[1] >= lrs[0]
    # Cosine tail: lrs decrease until the end.
    for prev, cur in zip(lrs[2:], lrs[3:]):
        assert cur <= prev + 1e-9
    # After ``max_epochs + 1`` steps the cosine reaches ``min_lr_ratio`` exactly.
    assert math.isclose(lrs[-1], 0.01, abs_tol=1e-5)


def test_warmup_cosine_rejects_invalid_arguments():
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError):
        WarmupCosineLR(optimizer, warmup_epochs=-1, max_epochs=10)
    with pytest.raises(ValueError):
        WarmupCosineLR(optimizer, warmup_epochs=10, max_epochs=10)
    with pytest.raises(ValueError):
        WarmupCosineLR(optimizer, warmup_epochs=1, max_epochs=10, min_lr_ratio=0.0)
