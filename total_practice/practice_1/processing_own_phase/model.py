from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .config import INPUT_DIM, NUM_CLASSES


class FashionMNISTModel(nn.Module):
    def __init__(
        self,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
        input_dim: int = INPUT_DIM,
    ):
        super().__init__()
        resolved_hidden_dims = (
            (128,) if hidden_dims is None else tuple(hidden_dims)
        )

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if any(
            hidden_dim <= 0
            for hidden_dim in resolved_hidden_dims
        ):
            raise ValueError("hidden dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the interval [0, 1)"
            )

        layers: List[nn.Module] = [nn.Flatten()]
        in_features = input_dim

        for hidden_dim in resolved_hidden_dims:
            layers.extend((
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
            ))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)
        self.hidden_dims = resolved_hidden_dims
        self.dropout = dropout
        self.num_classes = num_classes
        self.input_dim = input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


FashionMLP = FashionMNISTModel


def build_model(config: Dict) -> FashionMNISTModel:
    return FashionMNISTModel(
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        num_classes=config["num_classes"],
        input_dim=config["input_dim"],
    )


def sanity_check_model(
    model: FashionMNISTModel,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> Dict:
    sanity_model = FashionMNISTModel(
        hidden_dims=model.hidden_dims,
        dropout=model.dropout,
        num_classes=model.num_classes,
        input_dim=model.input_dim,
    )
    sanity_model.train()
    sanity_images = images[:8]
    sanity_labels = labels[:8]
    parameter = next(sanity_model.parameters())

    assert sanity_images.shape == (8, 1, 28, 28)
    assert sanity_images.dtype == parameter.dtype
    assert sanity_images.device == parameter.device
    assert sanity_labels.shape == (8,)
    assert sanity_labels.dtype == torch.int64

    logits = sanity_model(sanity_images)
    assert logits.shape == (8, sanity_model.num_classes)
    assert torch.isfinite(logits).all()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        sanity_model.parameters(),
        lr=0.01,
    )
    loss = criterion(logits, sanity_labels)
    assert torch.isfinite(loss)

    snapshot = {
        name: value.detach().clone()
        for name, value in sanity_model.named_parameters()
    }
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in sanity_model.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert any(
        gradient.abs().sum().item() > 0
        for gradient in gradients
    )

    optimizer.step()
    parameters_changed = any(
        not torch.equal(snapshot[name], parameter.detach())
        for name, parameter in sanity_model.named_parameters()
    )
    assert parameters_changed

    return {
        "output_shape": tuple(logits.shape),
        "loss": loss.item(),
        "gradients_finite": True,
        "parameters_changed": parameters_changed,
    }
