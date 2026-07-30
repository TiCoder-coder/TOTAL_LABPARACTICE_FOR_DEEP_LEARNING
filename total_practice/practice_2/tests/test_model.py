"""Test model wrapper."""

import torch
from processing_own_phase.model import build_model


def test_model_output_shape_all_models():
    """Test that all supported models produce the correct output shape [B, 10]."""
    for model_name in ["resnet18", "vgg16", "densenet121", "mobilenet_v3_small"]:
        model = build_model(model_name, "head_only", num_classes=10)
        dummy_input = torch.randn(2, 3, 224, 224)
        out = model(dummy_input)
        assert out.shape == (2, 10)


def test_model_summary():
    """Test the summary functionality."""
    model = build_model("mobilenet_v3_small", "head_only", num_classes=10)
    summary_dict = model.summary((2, 3, 224, 224))
    
    assert summary_dict["model_name"] == "mobilenet_v3_small"
    assert summary_dict["input_shape"] == (2, 3, 224, 224)
    assert summary_dict["output_shape"] == (2, 10)
    assert summary_dict["total_params"] > 0
    assert summary_dict["frozen_params"] > 0
    assert summary_dict["trainable_params"] > 0
    assert summary_dict["total_params"] == summary_dict["trainable_params"] + summary_dict["frozen_params"]


def test_freeze_modes():
    """Test that freeze modes correctly set requires_grad for ResNet and MobileNet."""
    # RESNET18
    model = build_model("resnet18", "head_only")
    assert not any(p.requires_grad for p in model.network.layer1.parameters())
    assert any(p.requires_grad for p in model.network.fc.parameters())

    model = build_model("resnet18", "partial_finetune")
    assert not any(p.requires_grad for p in model.network.layer1.parameters())
    assert any(p.requires_grad for p in model.network.layer4.parameters())
    
    # MOBILENET
    model = build_model("mobilenet_v3_small", "head_only")
    assert not any(p.requires_grad for p in model.network.features[0].parameters())
    assert any(p.requires_grad for p in model.network.classifier.parameters())
    
    model = build_model("mobilenet_v3_small", "partial_finetune")
    assert not any(p.requires_grad for p in model.network.features[0].parameters())
    # Should be unfrozen from block 12
    assert any(p.requires_grad for p in model.network.features[12].parameters())

    # Full finetune check
    model = build_model("resnet18", "full_finetune")
    assert all(p.requires_grad for p in model.network.parameters())
    assert model.training_mode == "full_finetune"
