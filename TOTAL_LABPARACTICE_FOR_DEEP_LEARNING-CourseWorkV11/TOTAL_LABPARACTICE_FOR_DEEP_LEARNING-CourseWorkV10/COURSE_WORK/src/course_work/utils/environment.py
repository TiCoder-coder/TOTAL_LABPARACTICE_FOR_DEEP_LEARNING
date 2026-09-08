import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch

from course_work.contracts.coursework import materialize_phase_0
from course_work.utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    get_project_root,
    read_json,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)
from course_work.utils.reproducibility import (
    DEVELOPMENT_SEED,
    configure_reproducibility,
    randomness_smoke_test,
    set_seed,
)


CORE_DISTRIBUTIONS = {
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit_learn": "scikit-learn",
    "matplotlib": "matplotlib",
    "jupyter": "jupyter",
    "ipykernel": "ipykernel",
}
COMPUTE_ENVIRONMENT_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)
STABLE_ENVIRONMENT_FIELDS = (
    "environment_id",
    "python_version",
    "python_executable",
    "pip_version",
    "package_versions",
    "platform",
    "platform_release",
    "architecture",
    "default_dtype",
    "development_seed",
    "deterministic_mode",
    "compute_environment_variables",
    "mps_fallback_enabled",
)
STABLE_KERNEL_FIELDS = (
    "kernel_name",
    "kernel_executable",
    "matches_interpreter",
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_kernel_contract(notebook_path: Path, interpreter: Path | None = None) -> dict[str, Any]:
    executable_path = interpreter or Path(sys.executable)
    executable = executable_path.resolve()
    notebook = read_json(notebook_path)
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    kernel_name = kernelspec.get("name")
    if not isinstance(kernel_name, str) or not kernel_name:
        return {"kernel_name": kernel_name, "kernel_spec_path": None, "kernel_executable": None, "matches_interpreter": False}
    candidates = [
        Path(sys.prefix) / "share/jupyter/kernels" / kernel_name / "kernel.json",
        executable_path.parent.parent / "share/jupyter/kernels" / kernel_name / "kernel.json",
        Path.home() / "Library/Jupyter/kernels" / kernel_name / "kernel.json",
        Path.home() / ".local/share/jupyter/kernels" / kernel_name / "kernel.json",
    ]
    kernel_path = next((path for path in candidates if path.is_file()), None)
    if kernel_path is None:
        return {"kernel_name": kernel_name, "kernel_spec_path": None, "kernel_executable": None, "matches_interpreter": False}
    kernel = read_json(kernel_path)
    argv = kernel.get("argv", [])
    command = argv[0] if argv else None
    if command == "python":
        kernel_executable = executable_path.parent / "python"
    elif isinstance(command, str):
        kernel_executable = Path(command)
    else:
        kernel_executable = None
    resolved_kernel = kernel_executable.resolve() if kernel_executable is not None and kernel_executable.exists() else kernel_executable
    return {
        "kernel_name": kernel_name,
        "kernel_spec_path": str(kernel_path),
        "kernel_executable": str(resolved_kernel) if resolved_kernel is not None else None,
        "matches_interpreter": resolved_kernel == executable,
    }


def package_versions() -> dict[str, str]:
    return {key: importlib.metadata.version(distribution) for key, distribution in CORE_DISTRIBUTIONS.items()}


def environment_inventory(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    notebook_path = root / "notebook_course_work/CourseWork.ipynb"
    selected_device = select_device()
    cuda_available = torch.cuda.is_available()
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    environment_variables = {name: os.environ[name] for name in COMPUTE_ENVIRONMENT_VARIABLES if name in os.environ}
    return {
        "environment_id": "ENV-v1",
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "pip_version": importlib.metadata.version("pip"),
        "package_versions": package_versions(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "working_directory": str(Path.cwd().resolve()),
        "project_root": str(root),
        "kernel": resolve_kernel_contract(notebook_path),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": cuda_device_name,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_device_name": "mps" if mps_available else None,
        "selected_device": str(selected_device),
        "default_dtype": str(torch.get_default_dtype()),
        "development_seed": DEVELOPMENT_SEED,
        "deterministic_mode": "D0",
        "compute_environment_variables": environment_variables,
        "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
    }


def environment_identity(inventory: dict[str, Any]) -> dict[str, Any]:
    identity = {field: inventory.get(field) for field in STABLE_ENVIRONMENT_FIELDS}
    kernel = inventory.get("kernel", {})
    identity["kernel"] = {field: kernel.get(field) for field in STABLE_KERNEL_FIELDS}
    return identity


def environment_identity_differences(
    recorded: dict[str, Any],
    current: dict[str, Any],
) -> tuple[str, ...]:
    recorded_identity = environment_identity(recorded)
    current_identity = environment_identity(current)
    return tuple(
        field
        for field in (*STABLE_ENVIRONMENT_FIELDS, "kernel")
        if recorded_identity[field] != current_identity[field]
    )


def device_smoke_test(device: torch.device | None = None) -> dict[str, Any]:
    selected_device = device or select_device()
    configure_reproducibility("D0")
    set_seed(DEVELOPMENT_SEED)
    x = torch.randn(4, 4, device=selected_device)
    y = torch.randn(4, 4, device=selected_device)
    z = x @ y
    tensor_ok = z.shape == (4, 4) and z.device.type == selected_device.type and bool(torch.isfinite(z).all().item())
    autograd_x = torch.randn(8, 4, device=selected_device, requires_grad=True)
    autograd_w = torch.randn(4, 1, device=selected_device, requires_grad=True)
    autograd_loss = (autograd_x @ autograd_w).pow(2).mean()
    autograd_loss.backward()
    gradients = (autograd_x.grad, autograd_w.grad)
    autograd_ok = bool(torch.isfinite(autograd_loss).item()) and all(
        gradient is not None and bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 1)).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.randn(16, 4, device=selected_device, dtype=torch.float32)
    targets = torch.randn(16, 1, device=selected_device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    module_loss = torch.nn.functional.mse_loss(predictions, targets)
    module_loss.backward()
    parameter_gradients = [parameter.grad for parameter in model.parameters()]
    gradient_ok = all(
        gradient is not None and bool(torch.isfinite(gradient).all().item()) for gradient in parameter_gradients
    )
    optimizer.step()
    module_ok = bool(torch.isfinite(module_loss).item()) and gradient_ok
    randomness = randomness_smoke_test()
    result = {
        "selected_device": str(selected_device),
        "imports_ok": True,
        "device_detected": True,
        "tensor_forward_ok": tensor_ok,
        "autograd_ok": autograd_ok,
        "module_training_step_ok": module_ok,
        "loss_finite": bool(torch.isfinite(module_loss).item()),
        "gradients_finite": gradient_ok,
        "seed_test_ok": randomness["status"] == "PASS",
        "randomness": randomness,
        "default_dtype": str(torch.get_default_dtype()),
    }
    required = [value for key, value in result.items() if key.endswith("_ok")]
    result["all_finite"] = result["loss_finite"] and result["gradients_finite"]
    result["status"] = "PASS" if all(required) and result["all_finite"] else "FAIL"
    return result


def dependency_freeze() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
    )
    content = completed.stdout
    for line in content.splitlines():
        lowered = line.lower()
        if "token=" in lowered or "password=" in lowered or "://" in line and "@" in line:
            raise RuntimeError("Dependency freeze contains a potentially sensitive direct reference")
    return content if content.endswith("\n") else f"{content}\n"


def load_validated_environment_signoff(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / "artifacts/environment/phase_1_signoff.json"
    if not signoff_path.is_file():
        raise FileNotFoundError(f"Phase 1 sign-off is missing: {signoff_path}")
    signoff = read_json(signoff_path)
    if signoff.get("phase_id") != 1 or signoff.get("phase_version") != "PHASE-1-v1":
        raise RuntimeError("Phase 1 sign-off identity is invalid")
    if signoff.get("status") != "PASS":
        raise RuntimeError("Phase 1 sign-off is not PASS")
    output_paths = signoff.get("output_paths")
    output_checksums = signoff.get("output_checksums")
    if not isinstance(output_paths, list) or not isinstance(output_checksums, dict):
        raise RuntimeError("Phase 1 output declaration is invalid")
    for relative_path in output_paths:
        path = (root / relative_path).resolve()
        path.relative_to(root)
        expected = output_checksums.get(relative_path)
        if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            raise RuntimeError(f"Phase 1 signed output is invalid: {relative_path}")
    return signoff


def materialize_phase_1(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_0 = materialize_phase_0(root)
    if phase_0.get("status") != "PASS":
        raise RuntimeError("Phase 0 sign-off is not PASS")
    environment_root = root / "artifacts/environment"
    environment_path = environment_root / "environment_report.json"
    freeze_path = environment_root / "requirements_freeze.txt"
    smoke_path = environment_root / "smoke_test_report.json"
    signoff_path = environment_root / "phase_1_signoff.json"
    phase_paths = (environment_path, freeze_path, smoke_path, signoff_path)
    existing_count = sum(path.exists() for path in phase_paths)
    if existing_count not in {0, len(phase_paths)}:
        raise RuntimeError("Phase 1 artifact set is incomplete")
    if existing_count == len(phase_paths):
        return load_validated_environment_signoff(root)
    inventory = environment_inventory(root)
    if not inventory["kernel"]["matches_interpreter"]:
        raise RuntimeError("Notebook kernel does not match the active interpreter")
    if inventory["default_dtype"] != "torch.float32":
        raise RuntimeError("Default torch dtype must be torch.float32")
    smoke = device_smoke_test()
    if smoke["status"] != "PASS":
        raise RuntimeError("Environment smoke test failed")
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    if not cuda_available and not mps_available:
        raise RuntimeError(
            "No GPU detected (no CUDA/MPS available). "
            "Training on CPU is prohibitively slow. "
            "Please ensure PyTorch with CUDA or MPS support is installed, "
            "or set the appropriate environment (e.g., activate a GPU-enabled venv)."
        )
    device_type = "cuda" if cuda_available else "mps"
    device_name = torch.cuda.get_device_name(0) if cuda_available else "Apple Silicon MPS"
    print(f"[ENV] GPU detected: {device_type} | {device_name}", flush=True)
    freeze = dependency_freeze()
    environment_report = {"created_at": datetime.now(timezone.utc).isoformat(), **inventory}
    write_json_once_or_verify(environment_path, environment_report)
    write_text_once_or_verify(freeze_path, freeze)
    write_json_once_or_verify(smoke_path, smoke)
    output_checksums = {
        "artifacts/environment/environment_report.json": sha256_file(environment_path),
        "artifacts/environment/requirements_freeze.txt": sha256_file(freeze_path),
        "artifacts/environment/smoke_test_report.json": sha256_file(smoke_path),
    }
    signoff = {
        "artifact_version": "ENV-v1",
        "phase_id": 1,
        "phase_version": "PHASE-1-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": "ENV-v1",
        "dataset_revision": None,
        "input_paths": ["artifacts/contracts/phase_0_signoff.json"],
        "input_checksums": {
            "artifacts/contracts/phase_0_signoff.json": sha256_file(root / "artifacts/contracts/phase_0_signoff.json")
        },
        "output_paths": list(output_checksums),
        "output_checksums": output_checksums,
        "config_fingerprint": phase_0["config_fingerprint"],
        "status": "PASS",
        "tests": [
            "interpreter_kernel_contract",
            "core_imports",
            "automatic_device_selection",
            "tensor_forward",
            "autograd_backward",
            "module_optimizer_step",
            "randomness_repeatability",
        ],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return signoff


def recover_environment_revision(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_0_path = root / "artifacts/contracts/phase_0_signoff.json"
    if not phase_0_path.is_file():
        raise RuntimeError("Phase 0 sign-off is missing")
    phase_0 = read_json(phase_0_path)
    if phase_0.get("status") != "PASS":
        raise RuntimeError("Phase 0 sign-off is not PASS")
    inventory = environment_inventory(root)
    if not inventory["kernel"]["matches_interpreter"]:
        raise RuntimeError("Notebook kernel does not match the active interpreter")
    if inventory["default_dtype"] != "torch.float32":
        raise RuntimeError("Default torch dtype must be torch.float32")
    if not inventory["cuda_available"] and not inventory["mps_available"]:
        raise RuntimeError("Environment recovery requires CUDA or MPS")
    smoke = device_smoke_test()
    if smoke["status"] != "PASS":
        raise RuntimeError("Environment recovery smoke test failed")
    freeze = dependency_freeze()
    environment_root = root / "artifacts/environment"
    environment_path = environment_root / "environment_report.json"
    freeze_path = environment_root / "requirements_freeze.txt"
    smoke_path = environment_root / "smoke_test_report.json"
    signoff_path = environment_root / "phase_1_signoff.json"
    targets = (environment_path, freeze_path, smoke_path, signoff_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    existing = [path for path in targets if path.exists()]
    archived_paths = []
    if existing:
        history_root = environment_root / "_history" / stamp
        history_root.mkdir(parents=True, exist_ok=False)
        for source in existing:
            destination = history_root / source.name
            source.replace(destination)
            archived_paths.append(str(destination.relative_to(root)))
    revision_id = f"ENV-R-{stamp}"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **inventory,
        "environment_revision_id": revision_id,
        "historical_environment_paths": archived_paths,
    }
    atomic_write_bytes(environment_path, canonical_json_bytes(report))
    atomic_write_bytes(freeze_path, freeze.encode("utf-8"))
    atomic_write_bytes(smoke_path, canonical_json_bytes(smoke))
    output_paths = [
        "artifacts/environment/environment_report.json",
        "artifacts/environment/requirements_freeze.txt",
        "artifacts/environment/smoke_test_report.json",
    ]
    signoff = {
        "artifact_version": "ENV-v1",
        "phase_id": 1,
        "phase_version": "PHASE-1-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": "ENV-v1",
        "environment_revision_id": revision_id,
        "dataset_revision": None,
        "input_paths": ["artifacts/contracts/phase_0_signoff.json"],
        "input_checksums": {
            "artifacts/contracts/phase_0_signoff.json": sha256_file(phase_0_path),
        },
        "output_paths": output_paths,
        "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        "config_fingerprint": phase_0.get("config_fingerprint"),
        "status": "PASS",
        "tests": [
            "interpreter_kernel_contract",
            "automatic_accelerator_selection",
            "device_smoke_test",
            "dependency_freeze",
            "historical_environment_preservation",
        ],
        "warnings": [],
        "discrepancies": [],
    }
    atomic_write_bytes(signoff_path, canonical_json_bytes(signoff))
    if read_json(environment_path).get("environment_revision_id") != revision_id:
        raise RuntimeError("Environment revision reload failed")
    if read_json(signoff_path).get("output_checksums") != signoff["output_checksums"]:
        raise RuntimeError("Environment revision sign-off reload failed")
    return signoff
