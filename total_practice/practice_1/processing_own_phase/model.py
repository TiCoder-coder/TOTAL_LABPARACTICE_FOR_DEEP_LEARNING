"""FashionMLP model class for FashionMNIST classification."""

from typing import List, Optional

import torch
import torch.nn as nn

from .config import INPUT_DIM, NUM_CLASSES


class FashionMLP(nn.Module):
    """Multi-Layer Perceptron for FashionMNIST classification.

    Args:
        hidden_dims: list of hidden layer sizes, e.g. [128] or [256, 128].
        dropout: dropout probability applied after each hidden layer.
        num_classes: number of output classes.
        input_dim: flattened input dimension (default 28*28=784).
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
        input_dim: int = INPUT_DIM,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128]

        layers: List[nn.Module] = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input: [B, 1, 28, 28] or [B, 784]
        Output: [B, num_classes] raw logits (no softmax).
        """
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.network(x)


def build_model(hidden_dims, dropout: float = 0.0) -> FashionMLP:
    """Factory function to build a FashionMLP."""
    return FashionMLP(hidden_dims=hidden_dims, dropout=dropout)


def sanity_check_model(model: FashionMLP, device: torch.device, batch_size: int = 64) -> dict:
    """Run quick sanity checks on the model.

    Verifies:
        1. Forward pass shape [B, num_classes].
        2. Loss is finite.
        3. Backward pass populates gradients.
        4. Optimizer step changes weights.
    """
    model = model.to(device)
    model.train()
    dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)
    dummy_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)

    logits = model(dummy_input)
    assert logits.shape == (batch_size, NUM_CLASSES), f"Bad shape: {logits.shape}"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss = criterion(logits, dummy_labels)
    assert torch.isfinite(loss).item(), "Loss is not finite"

    initial_param_snapshot = [p.detach().clone() for p in model.parameters()]

    optimizer.zero_grad()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()

    params_changed = any(
        not torch.equal(p.detach(), initial_param_snapshot[i])
        for i, p in enumerate(model.parameters())
    )

    return {
        "forward_shape": tuple(logits.shape),
        "loss_value": float(loss.item()),
        "loss_is_finite": torch.isfinite(loss).item(),
        "gradients_populated": has_grad,
        "params_changed": params_changed,
    }
