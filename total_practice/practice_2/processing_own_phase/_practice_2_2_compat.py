"""Bootstrap the consolidated Practice 2.2 package for legacy imports."""

import sys
from pathlib import Path


def enable_practice_2_2_import():
    src = Path(__file__).resolve().parents[2] / "practice_2_2" / "src"
    if not (src / "practice_2_2").is_dir():
        raise ImportError(f"Canonical Practice 2.2 package is missing: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

