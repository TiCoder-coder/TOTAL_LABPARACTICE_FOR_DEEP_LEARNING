import random
from typing import Any

import numpy as np
import torch


DEVELOPMENT_SEED = 42
FINAL_SEEDS = (42, 123, 2026)
REPRODUCIBILITY_MODES = ("D0", "D1")


def set_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def configure_reproducibility(mode: str = "D0") -> dict[str, Any]:
    if mode not in REPRODUCIBILITY_MODES:
        raise ValueError(f"mode must be one of {REPRODUCIBILITY_MODES}")
    strict = mode == "D1"
    torch.use_deterministic_algorithms(strict, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = strict
    return {
        "mode": mode,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
    }


def seed_worker(worker_id: int) -> None:
    if not isinstance(worker_id, int) or worker_id < 0:
        raise ValueError("worker_id must be a non-negative integer")
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_torch_generator(seed: int) -> torch.Generator:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def randomness_smoke_test(seed: int = DEVELOPMENT_SEED) -> dict[str, Any]:
    set_seed(seed)
    python_first = [random.random() for _ in range(4)]
    numpy_first = np.random.random(4)
    torch_first = torch.rand(4)
    generator_first = np.random.default_rng(seed).random(4)
    set_seed(seed)
    python_second = [random.random() for _ in range(4)]
    numpy_second = np.random.random(4)
    torch_second = torch.rand(4)
    generator_second = np.random.default_rng(seed).random(4)
    result = {
        "seed": seed,
        "python_random_repeatable": python_first == python_second,
        "numpy_random_repeatable": bool(np.array_equal(numpy_first, numpy_second)),
        "numpy_generator_repeatable": bool(np.array_equal(generator_first, generator_second)),
        "torch_random_repeatable": bool(torch.equal(torch_first, torch_second)),
    }
    result["status"] = "PASS" if all(value for key, value in result.items() if key.endswith("repeatable")) else "FAIL"
    return result
