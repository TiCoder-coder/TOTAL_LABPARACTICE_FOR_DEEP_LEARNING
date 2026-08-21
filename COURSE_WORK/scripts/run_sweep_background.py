"""Backwards-compatible shim that delegates to ``run_phase_background.py``.

Keeps the legacy ``run_sweep_background.py [--dry-run|--status]`` interface
working for callers (and older docs / scripts) while the unified launcher
takes over all dispatching.

Examples (legacy, still supported):
    python3 scripts/run_sweep_background.py --dry-run
    python3 scripts/run_sweep_background.py --status
    python3 scripts/run_sweep_background.py     # spawn the sweep chain (detached)
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here.parent))

    from scripts import run_phase_background as _unified  # type: ignore

    if "--status" in sys.argv[1:]:
        sys.exit(_unified.main(["--status"]))
    if "--dry-run" in sys.argv[1:]:
        sys.exit(_unified.main(["sweep", "--dry-run"]))
    # Anything else (including no args) means "launch the sweep".
    sys.exit(_unified.main(["sweep"]))
