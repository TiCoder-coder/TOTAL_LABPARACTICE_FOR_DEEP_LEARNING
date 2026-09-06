"""Phase 52 — main entry point.

Usage:
    python -m course_work.phase52.materialize [--project-root PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .orchestrator import run_phase52
from .sources import PHASE_DIR_REL


def main() -> dict[str, any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="COURSE_WORK project root",
    )
    args = parser.parse_args()
    root = args.project_root or Path("/Users/vientu/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
    t0 = time.time()
    result = run_phase52(root)
    elapsed = time.time() - t0
    print(f"Phase 52 extraction complete in {elapsed:.1f}s", file=sys.stderr)
    return result


if __name__ == "__main__":
    main()
