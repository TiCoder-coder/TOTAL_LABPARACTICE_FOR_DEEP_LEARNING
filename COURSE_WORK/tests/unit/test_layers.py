import pytest
import torch

from course_work.models.transformer_regressor import TransformerRegressor, validate_transformer_config
from course_work.sweeps.layers import CONDITIONS, inspect_layer_geometry
from course_work.utils.reproducibility import set_seed


def model_config() -> dict:
    return {
        "input_size": 33,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 128,
        "dropout": 0.1,
        "pooling": "LAST_STEP",
        "activation": "GELU",
    }


def test_layer_registry_is_exact() -> None:
    assert [(item.condition_id, item.num_layers, item.execution_mode) for item in CONDITIONS] == [
        ("N1", 1, "TRAIN_NEW"),
        ("N2", 2, "REUSE_REFERENCE"),
    ]


def test_layer_geometry_changes_only_depth_and_adds_independent_layer() -> None:
    comparison = inspect_layer_geometry(model_config())
    by_id = {item["condition_id"]: item for item in comparison["candidates"]}
    assert comparison["status"] == "PASS"
    assert comparison["config_delta"] == ["num_layers"]
    assert comparison["state_dict_subset"] is True
    assert comparison["shared_parameter_shapes_equal"] is True
    assert comparison["layers_independent"] is True
    assert comparison["parameter_delta"] > 0
    assert by_id["N1"]["trainable_parameter_count"] == 35713
    assert by_id["N2"]["trainable_parameter_count"] == 69185
    assert comparison["parameter_delta"] == 33472
    assert comparison["n2_only_key_numel"] == 33472
    assert comparison["parameter_delta"] == comparison["n2_only_key_numel"]
    assert all(row["status"] == "PASS" for row in comparison["state_key_rows"])
    assert all(row["status"] == "PASS" for row in comparison["shared_schema_rows"])
    assert all(row["status"] == "PASS" for row in comparison["independence_rows"])
    assert all(row["status"] == "PASS" for row in comparison["stack_rows"])
    assert len(comparison["stack_rows"]) == 4
    assert next(row for row in comparison["stack_rows"] if row["layer_id"] == "N1" and row["stack_index"] == 1)["present"] is False
    assert by_id["N1"]["prediction_shape"] == [2, 1]
    assert by_id["N2"]["prediction_shape"] == [2, 1]
    assert len(by_id["N1"]["attention_shapes"]) == 1
    assert len(by_id["N2"]["attention_shapes"]) == 2
    assert all(key.startswith("encoder.layers.1.") for key in comparison["extra_state_keys"])


def test_invalid_layer_count_is_rejected() -> None:
    invalid = model_config()
    invalid["num_layers"] = 0
    with pytest.raises(ValueError, match="num_layers"):
        validate_transformer_config(invalid)


@pytest.mark.parametrize("batch_size", [1, 7, 32])
def test_n1_forward_backward_and_partial_batch_contract(batch_size: int) -> None:
    config = model_config()
    config["num_layers"] = 1
    set_seed(42)
    model = TransformerRegressor(config)
    model.eval()
    sample = torch.randn(batch_size, 36, config["input_size"])
    prediction = model(sample)
    inspected_prediction, attention = model.forward_with_attention(sample)
    assert prediction.shape == (batch_size, 1)
    assert torch.isfinite(prediction).all()
    assert torch.allclose(prediction, inspected_prediction, atol=1e-6, rtol=1e-6)
    assert len(attention) == 1
    assert attention[0].shape == (batch_size, config["num_heads"], 36, 36)
    assert torch.isfinite(attention[0]).all()
    prediction.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_n2_attention_depth_contract_and_no_parameter_sharing() -> None:
    config = model_config()
    set_seed(42)
    model = TransformerRegressor(config)
    model.eval()
    sample = torch.randn(3, 36, config["input_size"])
    prediction, attention = model.forward_with_attention(sample)
    assert prediction.shape == (3, 1)
    assert len(attention) == 2
    assert all(value.shape == (3, config["num_heads"], 36, 36) for value in attention)
    assert all(torch.isfinite(value).all() and (value >= 0).all() for value in attention)
    layer0 = dict(model.encoder.layers[0].named_parameters())
    layer1 = dict(model.encoder.layers[1].named_parameters())
    assert set(layer0) == set(layer1)
    assert all(layer0[name] is not layer1[name] for name in layer0)
    assert all(layer0[name].data_ptr() != layer1[name].data_ptr() for name in layer0)
