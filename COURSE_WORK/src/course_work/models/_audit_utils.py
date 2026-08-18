from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from course_work.utils.artifacts import sha256_bytes, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def source_code_fingerprint(path: Path) -> str:
    return sha256_file(path)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def build_parameter_audit_rows(module: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, parameter in module.named_parameters():
        rows.append(
            {
                "parameter_name": name,
                "shape": "x".join(str(dim) for dim in parameter.shape),
                "numel": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
            }
        )
    return rows


def build_module_audit_rows(module: nn.Module, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, child in module.named_children():
        qualified = f"{prefix}.{name}" if prefix else name
        rows.append(
            {
                "module_path": qualified,
                "module_type": type(child).__name__,
                "trainable_parameters": count_trainable_parameters(child),
            }
        )
        if list(child.children()):
            rows.extend(build_module_audit_rows(child, qualified))
    return rows


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    detached = tensor.detach().cpu().contiguous()
    return sha256_bytes(detached.numpy().tobytes())


def assert_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {actual}")

