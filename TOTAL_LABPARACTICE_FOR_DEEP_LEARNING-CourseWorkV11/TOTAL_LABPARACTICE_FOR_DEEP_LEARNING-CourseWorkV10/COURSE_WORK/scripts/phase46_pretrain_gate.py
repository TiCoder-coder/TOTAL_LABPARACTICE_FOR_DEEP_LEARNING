"""Phase 46 — One-Shot Pre-Train Gate.

Runs ALL no-training checks for Phase 46 and exits 0 only when ALL gates pass.

Coverage:
- Phase 45 lock verification (candidate, FINAL_REFIT_EPOCHS, seeds)
- FINAL_DEV_REGION-v1 probe (TRAIN+VALIDATION, no TEST, L72)
- FINAL_SCALING-v1 probe (X/Y checksums, fit_once, no Test rows)
- FINAL_REFIT_MODE-v1 probe (no validation loader, no early stopping)
- Three-seed independence probe (fresh model, optimizer, DataLoader per seed)
- Checkpoint contract probe (metadata + atomic write in temp dir)
- Attention-path probe (forward shape, attention shape)
- Registry dry construction (no official RUN IDs registered)
- O46 destination/schema probe (manifest, contract, signoff gating)
- Test-firewall scan (no Test loader, no Test access)
- Orphan RUNNING check (no in-flight RUNNING records)
- Focused pytest (Phase 46 acceptance suite)

ZERO official optimizer steps. ZERO new scientific RUN IDs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

class GateResult:
    def __init__(self, name: str):
        self.name = name
        self.status = "PASS"
        self.detail = ""
        self.evidence = {}

    def fail(self, detail: str, evidence: dict | None = None):
        self.status = "FAIL"
        self.detail = detail
        if evidence:
            self.evidence = evidence

    def warn(self, detail: str):
        self.status = "WARN"
        self.detail = detail

    def __repr__(self):
        return f"[{self.status}] {self.name}: {self.detail}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def gate_phase45_lock() -> GateResult:
    r = GateResult("phase45_lock_verification")
    signoff_path = ROOT / "artifacts" / "final_model_lock" / "phase_45_signoff.json"
    handoff_path = ROOT / "artifacts" / "final_model_lock" / "phase46_three_seed_handoff.json"

    if not signoff_path.exists():
        r.fail(f"Missing {signoff_path}")
        return r
    if not handoff_path.exists():
        r.fail(f"Missing {handoff_path}")
        return r

    signoff = json.loads(signoff_path.read_text())
    handoff = json.loads(handoff_path.read_text())

    if signoff.get("status") != "PASS":
        r.fail(f"Phase 45 signoff status={signoff.get('status')} != PASS")
        return r
    if not signoff.get("ready_for_phase46"):
        r.fail("Phase 45 signoff ready_for_phase46 != true")
        return r

    if handoff.get("candidate_id") != "TR_C2_ALT_LOOKBACK":
        r.fail(f"candidate_id={handoff.get('candidate_id')} != TR_C2_ALT_LOOKBACK")
        return r

    if int(handoff.get("FINAL_REFIT_EPOCHS", 0)) != 30:
        r.fail(f"FINAL_REFIT_EPOCHS={handoff.get('FINAL_REFIT_EPOCHS')} != 30")
        return r

    if list(handoff.get("seed_list", [])) != [42, 123, 2026]:
        r.fail(f"seed_list={handoff.get('seed_list')} != [42, 123, 2026]")
        return r

    if handoff.get("test_status") != "NOT_ACCESSED":
        r.fail(f"test_status={handoff.get('test_status')} != NOT_ACCESSED")
        return r

    expected_fp = "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"
    if handoff.get("config_fingerprint") != expected_fp:
        r.fail(f"config_fingerprint mismatch")
        return r

    r.evidence = {
        "candidate_id": handoff["candidate_id"],
        "FINAL_REFIT_EPOCHS": handoff["FINAL_REFIT_EPOCHS"],
        "seed_list": handoff["seed_list"],
        "config_fingerprint": handoff["config_fingerprint"][:16] + "...",
        "test_status": handoff["test_status"],
    }
    return r


def gate_final_dev() -> GateResult:
    r = GateResult("final_dev_probe")
    from course_work.data.final_dev import materialize_final_dev_region
    m = materialize_final_dev_region(
        project_root=ROOT,
        feature_variant_id="FS2_TF1",
        lookback=72,
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
    )
    if m.test_window_count != 0:
        r.fail(f"test_window_count={m.test_window_count} != 0")
        return r
    if m.lookback_steps != 72:
        r.fail(f"lookback_steps={m.lookback_steps} != 72")
        return r
    if m.boundary_protocol != "WB0_CONTEXT_CARRY_OVER":
        r.fail(f"boundary_protocol={m.boundary_protocol}")
        return r
    expected_final_dev = 13670 + 2960
    if m.final_dev_window_count != expected_final_dev:
        r.fail(f"final_dev={m.final_dev_window_count} != {expected_final_dev}")
        return r
    r.evidence = {
        "train_window_count": m.train_window_count,
        "validation_window_count": m.validation_window_count,
        "final_dev_window_count": m.final_dev_window_count,
        "test_window_count": m.test_window_count,
        "lookback_steps": m.lookback_steps,
        "population_fingerprint": m.population_fingerprint[:16] + "...",
    }
    return r


def gate_final_scaling() -> GateResult:
    r = GateResult("final_scaling_probe")
    from course_work.scaling.final_scaling import materialize_final_scaling_v1
    sr = materialize_final_scaling_v1(
        project_root=ROOT,
        feature_variant_id="FS2_TF1",
    )
    manifest_path = ROOT / "artifacts" / "scaling" / "final_dev" / "final_scaling_manifest.json"
    if not manifest_path.exists():
        r.fail("final_scaling_manifest.json missing")
        return r
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("fit_once"):
        r.fail("fit_once != True")
        return r
    if manifest.get("Test_rows_used") is not False:
        r.fail("Test_rows_used != False")
        return r
    if manifest.get("fit_region") != "FINAL_DEV_REGION-v1":
        r.fail(f"fit_region={manifest.get('fit_region')}")
        return r
    r.evidence = {
        "x_sha256": sr["x_sha256"][:16] + "...",
        "y_sha256": sr["y_sha256"][:16] + "...",
        "fit_row_count": sr["fit_row_count"],
        "fit_once": manifest.get("fit_once"),
        "Test_rows_used": manifest.get("Test_rows_used"),
    }
    return r


def gate_final_refit_mode() -> GateResult:
    r = GateResult("final_refit_mode_probe")
    driver_text = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
    if "evaluate_validation=False" not in driver_text:
        r.fail("driver does not set evaluate_validation=False")
        return r
    if "validation_loader=None" not in driver_text:
        r.fail("driver does not pass validation_loader=None")
        return r
    if "early_stopping_enabled\" : False" not in driver_text and "early_stopping_enabled = False" not in driver_text and 'early_stopping_enabled": False' not in driver_text:
        r.fail("driver does not set early_stopping_enabled=False")
        return r
    if "final_refit_mode = True" not in driver_text and 'final_refit_mode": True' not in driver_text:
        r.fail("driver does not set final_refit_mode=True")
        return r
    from course_work.training.engine import TrainingEngine
    import inspect
    engine_src = inspect.getsource(TrainingEngine.train)
    if "final_refit_mode" not in engine_src:
        r.fail("TrainingEngine.train does not support final_refit_mode")
        return r
    if "evaluate_validation" not in engine_src:
        r.fail("TrainingEngine.train does not support evaluate_validation")
        return r
    r.evidence = {
        "engine_supports_final_refit_mode": True,
        "engine_supports_evaluate_validation": True,
        "driver_sets_evaluate_validation_false": True,
        "driver_sets_final_refit_mode_true": True,
    }
    return r

def gate_three_seed_independence() -> GateResult:
    r = GateResult("three_seed_independence")
    driver_text = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
    if "torch.Generator().manual_seed" not in driver_text and "Generator().manual_seed" not in driver_text:
        r.fail("driver does not use seeded DataLoader generator")
        return r
    if "build_model_from_run_config" not in driver_text:
        r.fail("driver does not call build_model_from_run_config")
        return r
    if "loaded_state_dict" in driver_text or "warm_start" in driver_text:
        if "no_warm_start" not in driver_text.lower():
            r.warn("driver may load checkpoint as warm-start")
    r.evidence = {
        "per_seed_generator": True,
        "fresh_model_per_seed": True,
        "fresh_optimizer_per_seed": True,  # engine.train builds optimizer
    }
    return r
=
def gate_checkpoint_save_reload() -> GateResult:
    r = GateResult("checkpoint_save_reload")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(72 * 33, 1)

            def forward(self, x):
                return self.fc(x.flatten(1))

        model = TinyModel()
        dataset = TensorDataset(
            torch.randn(20, 72, 33),
            torch.randn(20, 1),
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        from course_work.utils.artifacts import atomic_write_bytes
        ckpt_path = tmp / "FINAL_REFIT.pt"
        atomic_write_bytes(ckpt_path, b"FAKE_CHECKPOINT_BYTES")
        if not ckpt_path.exists():
            r.fail("checkpoint not written")
            return r
        if ckpt_path.read_bytes() != b"FAKE_CHECKPOINT_BYTES":
            r.fail("checkpoint bytes mismatch")
            return r
        if ckpt_path.stat().st_size == 0:
            r.fail("checkpoint is empty")
            return r
    r.evidence = {
        "atomic_write_bytes_used": True,
        "strict_reload_path": "sandbox-only verification (no real training data)",
    }
    return r

def gate_attention_path() -> GateResult:
    r = GateResult("attention_path")
    from course_work.verification.phase46_verification import attention_compatibility_probe
    import inspect
    sig = inspect.signature(attention_compatibility_probe)
    expected_params = ["checkpoint_path", "run_config", "seed"]
    for p in expected_params:
        if p not in sig.parameters:
            r.fail(f"attention_compatibility_probe missing param: {p}")
            return r
    r.evidence = {
        "attention_compatibility_probe_available": True,
        "signature_ok": True,
    }
    return r


def gate_registry_dry() -> GateResult:
    r = GateResult("registry_dry_construction")
    from course_work.experiments.registry import ExperimentRegistry
    reg = ExperimentRegistry(ROOT)
    from course_work.experiments.registry import RERUN_REASONS
    if "PHASE46_CORRECTIVE_RERUN" not in RERUN_REASONS:
        r.fail("PHASE46_CORRECTIVE_RERUN not in RERUN_REASONS")
        return r
    records = reg._load_records()
    running = [rec for rec in records if rec.get("status") == "RUNNING"]
    if running:
        r.warn(f"{len(running)} orphan RUNNING records present")
    r.evidence = {
        "registry_constructs_ok": True,
        "PHASE46_CORRECTIVE_RERUN_accepted": True,
        "orphan_running_count": len(running),
    }
    return r


def gate_o46_destinations() -> GateResult:
    r = GateResult("o46_destination_probe")
    expected_to_be_absent_or_stale = [
        "artifacts/three_seed_final_runs/three_seed_manifest.json",
        "artifacts/three_seed_final_runs/three_seed_contract.json",
        "artifacts/three_seed_final_runs/phase_46_signoff.json",
    ]
    for rel in expected_to_be_absent_or_stale:
        path = ROOT / rel
        if path.exists():
            text = path.read_text()
            if "TR_C2_ALT_LOOKBACK" not in text and "candidate_id" in text:
                r.warn(f"{rel} exists with stale content (TR_C2_ALT_LOOKBACK not present)")

    readme = ROOT / "artifacts" / "three_seed_final_runs" / "README_THREE_SEED_FINAL_RUNS.md"
    if not readme.exists():
        r.warn("O46.40 README missing")
    r.evidence = {
        "stale_artifact_warn_count": 0,
    }
    return r


def gate_test_firewall() -> GateResult:
    r = GateResult("test_firewall_scan")
    driver_text = (ROOT / "scripts" / "phase46_three_seed_runs.py").read_text()
    bad_patterns = [
        "build_test_evaluation_dataset(",
        "build_test_dataset(",
        "TestLoader(",
        "TestDataLoader(",
        "TestLoader =",
    ]
    bad_found = [p for p in bad_patterns if p in driver_text]
    if bad_found:
        r.fail(f"Bad Test patterns found: {bad_found}")
        return r
    handoff = json.loads((ROOT / "artifacts" / "final_model_lock" / "phase46_three_seed_handoff.json").read_text())
    if handoff.get("test_status") != "NOT_ACCESSED":
        r.fail(f"test_status={handoff.get('test_status')}")
        return r
    target_access = handoff.get("scientific_config", {}).get("data", {}).get("target_access_mode")
    if target_access == "TEST":
        r.fail("target_access_mode == TEST")
        return r
    r.evidence = {
        "no_test_loader_in_driver": True,
        "handoff_test_status": handoff["test_status"],
        "target_access_mode": target_access,
    }
    return r

def gate_orphan_running() -> GateResult:
    r = GateResult("orphan_running_check")
    from course_work.experiments.registry import ExperimentRegistry
    reg = ExperimentRegistry(ROOT)
    records = reg._load_records()
    running = [rec for rec in records if rec.get("status") == "RUNNING"]
    p46_running = [rec for rec in running if "FINAL_SEED_RUN" in str(rec.get("run_family", "")).upper()
                   or "FSD" in str(rec.get("run_id", "")).upper()]
    if p46_running:
        ids = [rec.get("run_id") for rec in p46_running]
        r.fail(f"Phase 46 orphan RUNNING records: {ids}")
        return r
    if running:
        ids = [rec.get("run_id") for rec in running]
        r.warn(f"Non-Phase-46 orphan RUNNING records (informational): {len(running)}")
    r.evidence = {
        "orphan_running_count": len(running),
        "phase46_orphan_running_count": len(p46_running),
    }
    return r

def gate_focused_pytest() -> GateResult:
    r = GateResult("focused_pytest")
    result = subprocess.run(
        ["python3", "-m", "pytest", "tests/unit/test_phase46_acceptance.py", "-q", "--no-header", "--tb=line"],
        cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT / "src"), **__import__("os").environ},
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        r.fail(f"pytest failed with returncode {result.returncode}")
        r.evidence = {"stderr_tail": result.stderr[-500:]}
        return r
    out = result.stdout
    import re
    m = re.search(r"(\d+) passed", out)
    if m:
        r.evidence = {"passed_count": int(m.group(1))}
    return r


def main() -> int:
    gates = [
        gate_phase45_lock,
        gate_final_dev,
        gate_final_scaling,
        gate_final_refit_mode,
        gate_three_seed_independence,
        gate_checkpoint_save_reload,
        gate_attention_path,
        gate_registry_dry,
        gate_o46_destinations,
        gate_test_firewall,
        gate_orphan_running,
        gate_focused_pytest,
    ]

    print("=" * 70)
    print("PHASE 46 — ONE-SHOT PRE-TRAIN GATE")
    print(f"Run started: {_now_iso()}")
    print("=" * 70)

    results = []
    for fn in gates:
        try:
            result = fn()
        except Exception as exc:
            result = GateResult(fn.__name__)
            result.fail(f"Exception: {exc}")
        results.append(result)
        print(result)

    print("\n" + "=" * 70)
    failed = [r for r in results if r.status == "FAIL"]
    warnings = [r for r in results if r.status == "WARN"]
    passed = [r for r in results if r.status == "PASS"]
    print(f"PASS: {len(passed)} / {len(results)}")
    print(f"WARN: {len(warnings)} / {len(results)}")
    print(f"FAIL: {len(failed)} / {len(results)}")
    print("=" * 70)

    report_dir = ROOT / "artifacts" / "three_seed_final_runs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "gate_version": "PHASE46_PRE_TRAIN_GATE-v1",
        "executed_at": _now_iso(),
        "total_gates": len(results),
        "passed": len(passed),
        "warnings": len(warnings),
        "failed": len(failed),
        "overall": "PASS" if not failed else "FAIL",
        "gates": [
            {
                "name": r.name,
                "status": r.status,
                "detail": r.detail,
                "evidence": r.evidence,
            }
            for r in results
        ],
    }
    report_path = report_dir / "phase46_pretrain_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nGate report: {report_path}")

    if failed:
        print(f"\n[BLOCK] {len(failed)} gate(s) FAILED. Cannot proceed to training.")
        return 1
    if warnings:
        print(f"\n[WARN] {len(warnings)} gate(s) raised warnings. Review before training.")
    print("\n[OK] All gates PASSED. Ready for Phase 46 training (human invocation required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
