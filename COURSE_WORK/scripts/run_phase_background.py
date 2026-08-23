"""Unified background launcher for all training-bearing phases.

Handles three runner kinds:
- ``phase <N>``       — single-shot baseline phases (20=LSTM, 21=Transformer B0, 22=Learning Diagnostics)
- ``sweep``           — sweep chain (Phase 23-32) via run_all_pending.py
- ``condition <S> <C>`` — single sweep condition (used internally by run_all_pending.py and for debugging)

All runners are spawned with ``start_new_session=True`` so the process group
survives notebook kernel shutdown. PID + status + stdout/stderr logs are
kept under ``artifacts/sweeps/`` for backward compatibility with
``run_sweep_background.py``.

Examples
--------
# Launch in the foreground of this shell (useful for debugging):
python3 scripts/run_phase_background.py phase 20 --foreground

# Detach and return immediately (notebook use):
python3 scripts/run_phase_background.py phase 20

# Sweep chain (all 13 pending conditions):
python3 scripts/run_phase_background.py sweep

# Inspect / cleanup:
python3 scripts/run_phase_background.py --status
python3 scripts/run_phase_background.py --stop
python3 scripts/run_phase_background.py --logs
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = ROOT / "scripts"
LOG_DIR = ROOT / "artifacts" / "sweeps" / "logs"
PID_FILE = ROOT / "artifacts" / "sweeps" / "sweep_runner.pid"
STATUS_FILE = LOG_DIR / "status.json"

# Resolved child-command specs. Each entry resolves to a (cmd_argv, cwd, log_prefix).
PHASE_RUNNERS = {
    # phase_id: (module_dotted_path, attr_name, optional_extra_argv)
    20: ("course_work.baselines.lstm_baseline", "materialize_phase_20", []),
    21: ("course_work.baselines.transformer_b0", "materialize_phase_21", []),
    22: ("course_work.diagnostics.learning_diagnostics", "materialize_phase_22", []),
}

# Dispatch for the legacy "sweep" subcommand.
SWEEP_CHILD = [sys.executable, str(SCRIPTS_DIR / "run_all_pending.py")]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status() -> dict[str, object]:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _write_status(payload: dict[str, object]) -> None:
    _ensure_dir(LOG_DIR)
    payload = {**payload, "updated_at": _now_iso()}
    STATUS_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _kill_existing() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        return False
    if not _is_alive(old_pid):
        PID_FILE.unlink(missing_ok=True)
        return False
    print(f"[run_phase_background] Killing existing runner PID={old_pid}...", flush=True)
    try:
        os.kill(old_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(1)
    if _is_alive(old_pid):
        try:
            os.kill(old_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)
    return True


def _resolve_phase_child(phase_id: int) -> list[str]:
    """Build the argv that will run a single phase materializer in a child interpreter."""
    if phase_id not in PHASE_RUNNERS:
        raise SystemExit(f"Unknown phase_id for background run: {phase_id}. "
                         f"Supported: {sorted(PHASE_RUNNERS)}")
    module_dotted, attr_name, extra_argv = PHASE_RUNNERS[phase_id]
    # Import check — fail fast in the launcher instead of inside the detached child.
    importlib.import_module(module_dotted)
    if not hasattr(sys.modules[module_dotted], attr_name):
        raise SystemExit(f"{module_dotted}.{attr_name} is not defined")
    # Bootstrap: cd into the project root then run the materialize function.
    # We import the module and call the function directly so the dispatcher
    # never relies on a missing ``if __name__ == "__main__"`` block.
    bootstrap = (
        "import sys; "
        "sys.path.insert(0, 'src'); "
        "from pathlib import Path; "
        f"from {module_dotted} import {attr_name}; "
        f"sys.exit(int({attr_name}(Path('.'))['status'] != 'PASS'))"
    )
    return [sys.executable, "-c", bootstrap, *extra_argv]


def _resolve_condition_child(sweep_id: str, condition_id: str) -> list[str]:
    runner = SCRIPTS_DIR / "run_single_condition.py"
    if not runner.exists():
        raise SystemExit(f"Missing condition runner: {runner}")
    return [sys.executable, str(runner), sweep_id, condition_id]


def _spawn_detached(child_argv: list[str], log_prefix: str, runner_kind: str,
                    runner_key: str) -> int:
    """Spawn child_argv in a fresh session and return its PID.

    The child is detached from the parent process group (``start_new_session=True``)
    so it survives notebook shutdown. stdout/stderr are appended to per-run log files.
    """
    _ensure_dir(LOG_DIR)
    _kill_existing()

    log_out = LOG_DIR / f"{log_prefix}.stdout.log"
    log_err = LOG_DIR / f"{log_prefix}.stderr.log"

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")

    proc = subprocess.Popen(
        child_argv,
        cwd=str(ROOT),
        stdout=open(str(log_out), "a", encoding="utf-8", buffering=1),
        stderr=open(str(log_err), "a", encoding="utf-8", buffering=1),
        start_new_session=True,
        env=env,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    _write_status({
        "status": "running",
        "pid": proc.pid,
        "runner_kind": runner_kind,
        "runner_key": runner_key,
        "child_argv": child_argv,
        "started_at": _now_iso(),
        "log_out": str(log_out.relative_to(ROOT)),
        "log_err": str(log_err.relative_to(ROOT)),
    })
    print(f"[run_phase_background] Started {runner_kind} {runner_key} PID={proc.pid}", flush=True)
    print(f"  stdout: {log_out}", flush=True)
    print(f"  stderr: {log_err}", flush=True)
    return proc.pid


def _run_foreground(child_argv: list[str]) -> int:
    """Run child_argv synchronously. Used for debugging or short phases."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    return subprocess.call(child_argv, cwd=str(ROOT), env=env)


def cmd_phase(args: argparse.Namespace) -> int:
    child_argv = _resolve_phase_child(args.phase_id)
    log_prefix = f"phase_{args.phase_id:02d}_background"
    if args.foreground:
        return _run_foreground(child_argv)
    _spawn_detached(child_argv, log_prefix, "phase", f"phase={args.phase_id:02d}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    child_argv = list(SWEEP_CHILD)
    if args.phase_id is not None:
        child_argv.extend(["--phase-id", str(args.phase_id)])
    if args.target_phase is not None:
        child_argv.extend(["--target-phase", str(args.target_phase)])
    if args.with_dependencies:
        child_argv.append("--with-dependencies")
    if args.audit_only:
        child_argv.append("--audit-only")
    if args.recover_environment:
        child_argv.append("--recover-environment")
    if args.dry_run:
        child_argv.append("--dry-run")
        print("Dry-run: would execute:", shlex.join(child_argv))
        return 0
    if args.foreground or args.audit_only:
        return _run_foreground(child_argv)
    selected_phase = args.target_phase or args.phase_id
    runner_key = "all_pending" if selected_phase is None else f"phase={selected_phase:02d}"
    log_prefix = "sweep_background" if selected_phase is None else f"phase_{selected_phase:02d}_resume"
    _spawn_detached(child_argv, log_prefix, "sweep", runner_key)
    return 0


def cmd_condition(args: argparse.Namespace) -> int:
    child_argv = _resolve_condition_child(args.sweep_id, args.condition_id)
    safe_id = f"{args.sweep_id}_{args.condition_id}"
    log_prefix = f"condition_{safe_id}"
    if args.foreground:
        return _run_foreground(child_argv)
    _spawn_detached(child_argv, log_prefix, "condition", safe_id)
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            alive = _is_alive(pid)
            print(f"PID file: {PID_FILE} (pid={pid}, alive={alive})", flush=True)
            if not alive:
                print("  stale — process no longer exists; safe to relaunch", flush=True)
        except ValueError:
            print(f"PID file corrupt: {PID_FILE}", flush=True)
    else:
        print("No PID file present.", flush=True)

    if STATUS_FILE.exists():
        status = _read_status()
        print(f"Status file: {STATUS_FILE}", flush=True)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    else:
        print("No status file present.", flush=True)
    return 0


def cmd_stop(_: argparse.Namespace) -> int:
    killed = _kill_existing()
    if killed:
        _write_status({"status": "stopped", "stopped_at": _now_iso()})
    else:
        print("[run_phase_background] No active runner to stop.", flush=True)
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    if not LOG_DIR.exists():
        print(f"No log directory: {LOG_DIR}", flush=True)
        return 1
    files = sorted(LOG_DIR.glob("*.log"))
    if not files:
        print(f"No logs in {LOG_DIR}", flush=True)
        return 0
    for path in files:
        print(f"\n===== {path} =====", flush=True)
        if args.tail and args.tail > 0:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                tail = fh.readlines()[-args.tail:]
                sys.stdout.write("".join(tail))
        else:
            print(path.read_text(encoding="utf-8", errors="replace"), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run training-bearing phases in the background.")
    parser.add_argument("--status", action="store_true",
                        help="Show the current background runner status and exit.")
    parser.add_argument("--stop", action="store_true",
                        help="Stop any running background runner.")
    parser.add_argument("--logs", action="store_true",
                        help="Show captured stdout/stderr logs.")
    parser.add_argument("--tail", type=int, default=0,
                        help="When combined with --logs, show only the last N lines per file.")

    sub = parser.add_subparsers(dest="command")

    p_phase = sub.add_parser("phase", help="Run a single-shot baseline phase (20, 21, 22).")
    p_phase.add_argument("phase_id", type=int, choices=sorted(PHASE_RUNNERS),
                         help="Phase ID to materialize.")
    p_phase.add_argument("--foreground", action="store_true",
                         help="Run in this process instead of spawning a detached child.")
    p_phase.set_defaults(func=cmd_phase)

    p_sweep = sub.add_parser("sweep", help="Run pending sweep conditions (Phase 23-34).")
    p_sweep.add_argument("--dry-run", action="store_true",
                         help="Print what would run without spawning.")
    p_sweep.add_argument("--phase-id", type=int, choices=tuple(range(23, 35)),
                         help="Resolve and run only missing conditions for one sweep Phase.")
    p_sweep.add_argument("--target-phase", type=int, choices=tuple(range(23, 35)))
    p_sweep.add_argument("--with-dependencies", action="store_true")
    p_sweep.add_argument("--audit-only", action="store_true")
    p_sweep.add_argument("--foreground", action="store_true")
    p_sweep.add_argument("--recover-environment", action="store_true")
    p_sweep.set_defaults(func=cmd_sweep)

    p_cond = sub.add_parser("condition", help="Run one sweep condition (debug; production uses 'sweep').")
    p_cond.add_argument("sweep_id")
    p_cond.add_argument("condition_id")
    p_cond.add_argument("--foreground", action="store_true")
    p_cond.set_defaults(func=cmd_condition)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.status:
        return cmd_status(args)
    if args.stop:
        return cmd_stop(args)
    if args.logs:
        return cmd_logs(args)
    if not args.command:
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
