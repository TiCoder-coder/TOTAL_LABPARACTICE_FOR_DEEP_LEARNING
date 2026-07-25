from pathlib import Path
from typing import Dict, Optional

import torch

from .config import MODEL_SAVE_PATH
from .model import FashionMNISTModel


def save_checkpoint(
    model: FashionMNISTModel,
    model_config: Dict,
    training_config: Dict,
    metadata: Dict,
    path: str = str(MODEL_SAVE_PATH),
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value
            in model.state_dict().items()
        },
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "metadata": dict(metadata),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: str = str(MODEL_SAVE_PATH),
    device: Optional[torch.device] = None,
) -> Dict:
    map_location = device if device is not None else "cpu"
    return torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )


def load_model_from_checkpoint(
    path: str = str(MODEL_SAVE_PATH),
    device: Optional[torch.device] = None,
) -> FashionMNISTModel:
    checkpoint = load_checkpoint(path, device=device)
    model = FashionMNISTModel(
        **checkpoint["model_config"],
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    if device is not None:
        model.to(device)
    model.eval()
    return model


def verify_loaded_model(
    original_model: FashionMNISTModel,
    loaded_model: FashionMNISTModel,
    inputs: torch.Tensor,
) -> Dict:
    original_model.eval()
    loaded_model.eval()

    with torch.inference_mode():
        original_logits = original_model(inputs)
        loaded_logits = loaded_model(inputs)

    predictions_match = torch.equal(
        original_logits.argmax(dim=1),
        loaded_logits.argmax(dim=1),
    )
    logits_match = torch.allclose(
        original_logits,
        loaded_logits,
    )
    maximum_logit_difference = (
        original_logits - loaded_logits
    ).abs().max().item()

    return {
        "predictions_match": bool(predictions_match),
        "logits_match": bool(logits_match),
        "maximum_logit_difference": float(
            maximum_logit_difference
        ),
    }
