"""Notebook-side helpers for ``scripts/run_phase_background.py``.

Importable from Jupyter cells without coupling the notebook to subprocess plumbing::

    from scripts.run_phase_background import (
        launch_phase, launch_sweep, launch_condition,
        poll_status, wait_for, stop, tail_logs,
    )

All helpers are thin wrappers over subprocess calls into the launcher script —
they never inline the launcher logic, so behavior stays single-sourced.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "run_phase_background.py"
LOG_DIR = ROOT / "artifacts" / "sweeps" / "logs"
STATUS_FILE = LOG_DIR / "status.json"
PID_FILE = ROOT / "artifacts" / "sweeps" / "sweep_runner.pid"


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(LAUNCHER), *args]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=check)


def launch_phase(phase_id: int, *, foreground: bool = False) -> subprocess.CompletedProcess[str]:
    """Spawn a baseline phase (20/21/22) in a detached process group.

    Default: detached (survives kernel shutdown).
    Set ``foreground=True`` only for debugging inside a notebook cell that
    wants to block until training finishes.
    """
    argv = ["phase", str(phase_id)]
    if foreground:
        argv.append("--foreground")
    return _run(argv)


def launch_sweep(*, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    """Spawn the sweep chain (Phase 23-30) in a detached process group."""
    argv = ["sweep"]
    if dry_run:
        argv.append("--dry-run")
    return _run(argv)


def launch_condition(sweep_id: str, condition_id: str, *, foreground: bool = False) -> subprocess.CompletedProcess[str]:
    argv = ["condition", sweep_id, condition_id]
    if foreground:
        argv.append("--foreground")
    return _run(argv)


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def poll_status() -> dict[str, Any]:
    """Return the most recent status payload plus liveness information.

    Keys:
        pid          — PID stored in artifacts/sweeps/sweep_runner.pid (or None)
        alive        — bool, whether os.kill(pid, 0) succeeded
        status       — raw status.json payload, or {} when missing
        recommended  — one of {"idle", "running", "stale", "unknown"}
    """
    pid = _read_pid()
    alive = _pid_alive(pid) if pid is not None else False
    status: dict[str, Any] = {}
    if STATUS_FILE.exists():
        try:
            status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {"error": "status.json invalid JSON"}

    if pid is None and not status:
        recommended = "idle"
    elif pid is None and status.get("status") == "running":
        recommended = "stale"
    elif not alive and status.get("status") == "running":
        recommended = "stale"
    elif alive:
        recommended = "running"
    else:
        recommended = "idle"

    return {
        "pid": pid,
        "alive": alive,
        "status": status,
        "recommended": recommended,
    }


def wait_for(states: tuple[str, ...] = ("stopped",), *,
             poll_seconds: float = 5.0, timeout_seconds: float | None = None) -> dict[str, Any]:
    """Block until the runner reports one of ``states`` (or disappears).

    Useful in a notebook cell when you *want* to keep the cell busy
    while the detached process does the real work. ``states`` should match
    the ``status`` field written by the launcher (e.g. ``"stopped"``).
    """
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    while True:
        snapshot = poll_status()
        current = snapshot["status"].get("status") if isinstance(snapshot["status"], dict) else None
        if snapshot["recommended"] in {"idle", "stale"} or current in states:
            return snapshot
        if deadline is not None and time.monotonic() >= deadline:
            return {**snapshot, "timed_out": True}
        time.sleep(poll_seconds)


def stop() -> subprocess.CompletedProcess[str]:
    """Ask the launcher to SIGTERM (then SIGKILL) the active runner."""
    return _run(["--stop"])


def tail_logs(name_glob: str = "*.log", lines: int = 40) -> dict[str, list[str]]:
    """Return the last ``lines`` of each log file matching ``name_glob``."""
    if not LOG_DIR.exists():
        return {}
    out: dict[str, list[str]] = {}
    for path in sorted(LOG_DIR.glob(name_glob)):
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            tail = fh.readlines()[-lines:]
        out[str(path.relative_to(ROOT))] = tail
    return out


__all__ = [
    "LAUNCHER",
    "LOG_DIR",
    "PID_FILE",
    "STATUS_FILE",
    "launch_phase",
    "launch_sweep",
    "launch_condition",
    "poll_status",
    "wait_for",
    "stop",
    "tail_logs",
]
