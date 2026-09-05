"""Pytest configuration for COURSE_WORK tests.

Adds the source directory to sys.path so tests can import course_work.* modules.
Also chdirs into COURSE_WORK so get_project_root() returns the right path.
"""

import os
import sys
from pathlib import Path

# COURSE_WORK root = parent of this conftest.py (tests/ is inside COURSE_WORK/)
COURSE_WORK_ROOT = Path(__file__).resolve().parent.parent

# Add src/ to sys.path
SRC = COURSE_WORK_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Change working directory to COURSE_WORK so get_project_root() works correctly
# (get_project_root uses Path(__file__).parents[3] which goes ABOVE COURSE_WORK)
os.environ.setdefault("COURSE_WORK_ROOT", str(COURSE_WORK_ROOT))
os.chdir(str(COURSE_WORK_ROOT))
