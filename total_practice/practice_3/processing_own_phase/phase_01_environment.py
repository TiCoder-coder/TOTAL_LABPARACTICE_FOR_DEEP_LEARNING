import os
import sys
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from importlib.metadata import version as get_version, PackageNotFoundError

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config import PROJECT_ROOT, RESULT_DIR, REQUIRED_PACKAGE_VERSIONS


# Use package versions from config.py (single source of truth)
REQUIRED_PACKAGES = REQUIRED_PACKAGE_VERSIONS


def ensure_directories() -> None:
    """
    Create required log and result directories if they don't exist.
    Uses PROJECT_ROOT from config to ensure correct absolute paths.
    """
    directories = [
        RESULT_DIR,
        PROJECT_ROOT / "docs" / "plan-doc" / "analysis_error",
        PROJECT_ROOT / "docs" / "plan-doc" / "plan_before_process",
        PROJECT_ROOT / "docs" / "plan-doc" / "plan_to_refactor&fix",
        PROJECT_ROOT / "docs" / "save_process_proceduce_own_phase_refactor&fix",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """
    Set global seed for reproducibility across random, numpy, and torch.
    If CUDA is available, sets cudnn deterministic and disables benchmark.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """
    Detect and return the available device.
    Priority: cuda -> mps (Apple Silicon) -> cpu.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_required_packages() -> Dict[str, Dict[str, str]]:
    """
    Check installed package versions against requirements.
    - Version mismatch: emits UserWarning (does not stop execution)
    - Missing package: raises Exception after checking all packages
    """
    results = {}
    missing_packages = []

    for package, required_version in REQUIRED_PACKAGES.items():
        try:
            installed_version = get_version(package)
            status = "OK" if installed_version == required_version else "WARNING"
            if status == "WARNING":
                warnings.warn(
                    f"Package {package} version {installed_version} does not match required {required_version}",
                    UserWarning,
                )
            results[package] = {
                "installed": installed_version,
                "required": required_version,
                "status": status,
            }
        except PackageNotFoundError:
            results[package] = {
                "installed": "MISSING",
                "required": required_version,
                "status": "ERROR",
            }
            missing_packages.append(package)

    if missing_packages:
        raise Exception(
            f"Missing required packages: {missing_packages}. "
            f"Run: pip install {' '.join(missing_packages)}"
        )

    return results


def check_huggingface_connectivity(checkpoint: str = "distilbert-base-uncased") -> Dict[str, Any]:
    """
    Attempt to load a lightweight tokenizer from Hugging Face Hub.
    Returns status dict: {'status': 'ok'/'failed', 'error': ...}
    """
    if not TRANSFORMERS_AVAILABLE:
        return {
            "status": "failed",
            "error": "transformers is not installed",
            "checkpoint": checkpoint,
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        _ = tokenizer("Test connectivity")
        return {
            "status": "ok",
            "error": None,
            "checkpoint": checkpoint,
            "vocab_size": tokenizer.vocab_size,
        }
    except Exception as error:
        return {
            "status": "failed",
            "error": str(error),
            "checkpoint": checkpoint,
        }


def print_environment_info() -> Dict[str, Any]:
    """
    Print and return comprehensive environment information.
    Includes Python version, device, seed, package status, and Hugging Face connectivity.
    """
    ensure_directories()

    device = get_device()
    packages_status = check_required_packages()
    hf_connectivity = check_huggingface_connectivity()

    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "device": device,
        "seed": 42,
        "packages": packages_status,
        "huggingface_connectivity": hf_connectivity,
    }

    print("\n" + "=" * 70)
    print("ENVIRONMENT INFORMATION")
    print("=" * 70)
    print(f"Timestamp: {info['timestamp']}")
    print(f"Python Version: {info['python_version']}")
    print(f"Device: {device}")
    print(f"Global Seed: {info['seed']}")
    print("\n--- Package Versions ---")
    for pkg, data in packages_status.items():
        status = data["status"]
        installed = data["installed"]
        required = data["required"]
        print(f"[{status:7}] {pkg:15} installed: {installed:12} required: {required}")

    print("\n--- Hugging Face Connectivity ---")
    hf_status = hf_connectivity["status"]
    print(f"Status: {hf_status}")
    if hf_status == "ok":
        print(f"  Checkpoint: {hf_connectivity['checkpoint']}")
        print(f"  Vocab size: {hf_connectivity['vocab_size']}")
    else:
        print(f"  Error: {hf_connectivity.get('error', 'Unknown error')}")
    print("=" * 70 + "\n")

    return info


def save_environment_report(info: Dict[str, Any]) -> Path:
    """
    Save environment info dictionary to a JSON file in docs/result/.
    Uses a fixed filename (overwrites on each run) as specified in the Phase 1 plan.
    Returns the path to the saved file.
    """
    ensure_directories()
    filepath = RESULT_DIR / "2026-08-10_phase01-environment-log.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)

    print(f"Environment report saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    set_seed(42)
    info = print_environment_info()
    save_environment_report(info)