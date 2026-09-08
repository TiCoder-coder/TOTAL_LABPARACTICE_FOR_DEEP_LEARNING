from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

# Paths
from course_workutils.artifacts import get_project_root


SEED_LIST = ("42", "123", "2026")
SEED_COLORS = {"42": "#1f77b4", "123": "#2ca02c", "2026": "#d62728"}
SEED_LABELS = {"42": "Seed 42", "123": "Seed 123", "2026": "Seed 2026"}
FIGURES_DIR_REL = Path("artifacts/residual_analysis/figures")


# ---------- Safety helpers ----------

def _safe_for_json(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return None
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_safe_for_json(a) for a in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _safe_for_json(v) for k, v in x.items()}
    if isinstance(x, (set, tuple, list)):
        return [_safe_for_json(a) for a in x]
    return str(x)


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _ensure_figures_dir(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    out = root / FIGURES_DIR_REL
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_figure(fig, project_root: Path | None, filename: str) -> Path:
    out_dir = _ensure_figures_dir(project_root)
    target = out_dir / _sanitize_filename(filename)
    fig.savefig(target, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return target
