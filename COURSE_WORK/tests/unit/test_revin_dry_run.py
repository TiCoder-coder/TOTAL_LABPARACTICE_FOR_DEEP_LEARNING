"""Phase 40 — dry-run does not call TrainingEngine.train() and does not access Test.

Tests prove the S18 dispatch path's `--dry-run` does NOT execute scientific training
and does NOT access Test data.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path('/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK')
CLI = ROOT / "scripts" / "run_single_condition.py"


@pytest.fixture(scope="module")
def s18_rn1_dry_run_result():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MPLCONFIGDIR"] = "/tmp/course-work-mpl-cache"
    result = subprocess.run(
        [sys.executable, "-u", str(CLI), "S18_REVIN", "RN1", "--dry-run"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result


@pytest.fixture(scope="module")
def s18_rn0_dry_run_result():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MPLCONFIGDIR"] = "/tmp/course-work-mpl-cache"
    result = subprocess.run(
        [sys.executable, "-u", str(CLI), "S18_REVIN", "RN0", "--dry-run"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result


@pytest.fixture(scope="module")
def s17_dry_run_result():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MPLCONFIGDIR"] = "/tmp/course-work-mpl-cache"
    result = subprocess.run(
        [sys.executable, "-u", str(CLI), "S17_GRADIENT_CLIPPING", "GC0", "--dry-run"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result


def test_s18_rn1_dry_run_passes(s18_rn1_dry_run_result):
    assert s18_rn1_dry_run_result.returncode == 0, s18_rn1_dry_run_result.stdout + s18_rn1_dry_run_result.stderr
    assert "GUARDED SMOKE COMPLETE" in s18_rn1_dry_run_result.stdout
    assert "[25/25]" in s18_rn1_dry_run_result.stdout
    assert "before TrainingEngine.train()" in s18_rn1_dry_run_result.stdout


def test_s18_rn1_dry_run_no_test_access(s18_rn1_dry_run_result):
    out = s18_rn1_dry_run_result.stdout
    # Test firewall must be active
    assert "Test firewall: PASS" in out
    assert "No Test data accessed" in out


def test_s18_rn1_dry_run_still_succeeds_without_persisting_run(s18_rn1_dry_run_result):
    """The dry-run path (--dry-run flag) must still stop before TrainingEngine.train()
    and must NOT create a new scientific run. The existing RN1 scientific run on disk
    was created by a separate, human-authorized training invocation (canonical CLI
    without --dry-run), not by the dry-run path."""
    out = s18_rn1_dry_run_result.stdout
    assert "No scientific run created" in out
    assert "No checkpoint" in out


def test_s18_rn0_resolves_to_reuse_reference(s18_rn0_dry_run_result):
    assert s18_rn0_dry_run_result.returncode == 0, s18_rn0_dry_run_result.stdout + s18_rn0_dry_run_result.stderr
    out = s18_rn0_dry_run_result.stdout
    assert "mode=REUSE_REFERENCE" in out
    assert "reference=RUN_TR_S14_0023_A711A9B8" in out


def test_s17_dry_run_still_works(s17_dry_run_result):
    # S17 GC0 must continue to PASS after our changes
    assert s17_dry_run_result.returncode == 0, s17_dry_run_result.stdout + s17_dry_run_result.stderr
    out = s17_dry_run_result.stdout
    assert "S17_GRADIENT_CLIPPING" in out
    assert "GUARDED SMOKE COMPLETE" in out
    assert "Registry/config validation: PASS" in out


def test_unknown_sweep_dry_run_still_acknowledged():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MPLCONFIGDIR"] = "/tmp/course-work-mpl-cache"
    result = subprocess.run(
        [sys.executable, "-u", str(CLI), "S99_UNKNOWN", "XYZ", "--dry-run"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "Dry-run only implemented" in result.stdout