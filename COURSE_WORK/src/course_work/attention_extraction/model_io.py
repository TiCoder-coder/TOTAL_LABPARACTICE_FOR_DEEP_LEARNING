"""Phase 52 — strict model loading + extraction core.

EXTRACTION + INTEGRITY ONLY.

Provides:

  - strict_load_checkpoint(seed) → TransformerRegressor (eval mode)
  - model_fingerprint(model) → SHA256 of state_dict (for mutation audit)
  - extract_batch(model, x) → (pred, attention_layers)
      attention_layers is a list of [B, H, L, L] tensors, NO averaging.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch

from course_work.models.transformer_regressor import (
    TransformerRegressor,
    TransformerModelConfig,
)


def build_model_from_final_lock(project_root: Path) -> TransformerRegressor:
    """Build the exact TransformerRegressor from frozen final lock config.

    Reuses the same dataclass used by Phase 16 / 17 / 46 / 47. No mutation.
    """
    cfg_fp = project_root / "artifacts/final_model_lock/final_model_scientific_config.json"
    import json

    cfg_doc = json.loads(cfg_fp.read_text())

    model_doc = cfg_doc["model"]
    cfg = TransformerModelConfig(
        model_family=model_doc["model_family"],
        model_version=model_doc["model_version"],
        implementation_version=model_doc["implementation_version"],
        attention_aware=bool(model_doc["attention_aware"]),
        input_size=int(model_doc["input_size"]),
        d_model=int(model_doc["d_model"]),
        num_heads=int(model_doc["num_heads"]),
        num_layers=int(model_doc["num_layers"]),
        ffn_dim=int(model_doc["ffn_dim"]),
        dropout=float(model_doc["dropout"]),
        pooling=model_doc["pooling"],
        output_size=int(model_doc["output_size"]),
        activation=model_doc["activation"],
        positional_encoding_type=model_doc["positional_encoding_type"],
        norm_first=bool(model_doc["norm_first"]),
    )

    model = TransformerRegressor(cfg)
    return model


def strict_load_checkpoint(
    seed: int,
    checkpoint_path: Path,
    model: TransformerRegressor,
) -> dict[str, int]:
    """Strict-load the frozen final checkpoint into the model.

    Strict-load contract:
      - weights_only=True (or strict dict load with no allowlist gaps)
      - missing_keys == 0
      - unexpected_keys == 0

    Returns dict with keys: missing_keys, unexpected_keys.
    """
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    # Some checkpoints store {"state_dict": ...}, {"model_state_dict": ...}, or the dict directly.
    if isinstance(state, dict):
        for wrapper_key in ("model_state_dict", "state_dict"):
            if wrapper_key in state and isinstance(state[wrapper_key], dict):
                sd = state[wrapper_key]
                break
        else:
            sd = state
    else:
        sd = state

    result = model.load_state_dict(sd, strict=True)

    missing_keys = len(result.missing_keys)
    unexpected_keys = len(result.unexpected_keys)

    if missing_keys != 0 or unexpected_keys != 0:
        raise RuntimeError(
            f"Strict load failed for seed {seed}: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )

    model.eval()
    return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}


def model_state_fingerprint(model: TransformerRegressor) -> str:
    """SHA256 of the model state_dict (for mutation audit)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


@torch.inference_mode()
def extract_batch(
    model: TransformerRegressor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run inspection forward pass.

    Returns (prediction, attention_layers).
      prediction: [B, 1]
      attention_layers[i]: [B, H, L, L] float32 (per layer, per head)

    No averaging. Returns raw per-head, per-layer attention.
    """
    pred, attention_layers = model.forward_with_attention(x)
    return pred, attention_layers
