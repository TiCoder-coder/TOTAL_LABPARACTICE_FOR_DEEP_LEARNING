"""Pytest configuration for COURSE_WORK tests.

Adds the source directory to sys.path so tests can import course_work.* modules.
Also chdirs into COURSE_WORK so get_project_root() returns the right path.
"""

import os
import sys
from pathlib import Path

COURSE_WORK_ROOT = Path(__file__).resolve().parent.parent

SRC = COURSE_WORK_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("COURSE_WORK_ROOT", str(COURSE_WORK_ROOT))
os.chdir(str(COURSE_WORK_ROOT))
