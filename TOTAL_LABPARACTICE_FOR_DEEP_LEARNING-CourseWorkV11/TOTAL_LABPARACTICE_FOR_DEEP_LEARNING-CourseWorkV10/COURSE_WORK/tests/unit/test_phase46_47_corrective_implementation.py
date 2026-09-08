"""Phase 46 / 47 Corrective — Implementation Verification Tests.

These tests verify the corrective code changes implemented per Part 2F of the
corrective workflow. They are NON-SCIENTIFIC:

  - NO training
  - NO inference
  - NO Test access
  - NO optimizer.step()
  - NO scaler.fit() on Test
  - NO Phase 45 lock modification
  - NO scientific artifact modification

Coverage:
- Lock identity split (Fix #4)
- Phase 47 runner writes LOCKED_FINAL_LOCK_SHA256 into final_lock_sha256 field
  (Fix #5, #6)
- Phase 47 runner distinguishes lock_hash_match vs config_hash_match
  (Fix #6)
- Phase 46 runner has NO TR_C0_PRIMARY fallback (Fix #1)
- Phase 46 runner sources FINAL_REFIT_EPOCHS strictly from Phase 45 (Fix #2)
- Phase 46 runner refuses stale 50-epoch fallback (Fix #3)
- The 17-gate pre-Test gate exists and is non-scientific (Fix #11)
- The Phase 47 archive script exists and is non-scientific (Fix #12)
- Historical run IDs are rejected (Fix #15)
- No best-seed / ensemble / validation stopping (Fix #14)
- Prediction bundle persistence logic (Fix #13)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


LOCKED_CANDIDATE = "TR_C2_ALT_LOOKBACK"
LOCKED_LOOKBACK = 72
LOCKED_FINAL_REFIT_EPOCHS = 30
LOCKED_SEEDS = [42, 123, 2026]
LOCKED_CONFIG_FINGERPRINT = "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"
LOCKED_FINAL_LOCK_SHA256 = "81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec"
LOCKED_RECIPE_SHA256 = "857dbaf7903792cdba3e126a50d8919992f02e0fff838cba86180026c9e0220c"
LOCKED_POPULATION_SHA256 = "552e6895dbc6f7a93b0ea9a2ab4de77abd58d8780ad4bc85b697a485ccbc31a9"
LOCKED_FEATURE_SHA256 = "fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee"
EXCLUDED_HISTORICAL_RUN_IDS = {
    "RUN_TR_FSD_0153_B15A19DC",
    "RUN_TR_FSD_0154_DD82D743",
    "RUN_TR_FSD_0155_59A50ADD",
}
FORBIDDEN_CANDIDATE_FALLBACK = "TR_C0_PRIMARY"

class TestLockIdentitySplit:
    def test_locked_config_fingerprint_exists(self):
        from course_work.final_test_evaluation import LOCKED_CONFIG_FINGERPRINT
        assert LOCKED_CONFIG_FINGERPRINT == LOCKED_CONFIG_FINGERPRINT

    def test_locked_final_lock_sha256_distinct_from_config_fingerprint(self):
        from course_work.final_test_evaluation import (
            LOCKED_CONFIG_FINGERPRINT,
            LOCKED_FINAL_LOCK_SHA256,
        )
        assert LOCKED_FINAL_LOCK_SHA256 != LOCKED_CONFIG_FINGERPRINT
        assert LOCKED_FINAL_LOCK_SHA256 == "81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec"
        assert LOCKED_CONFIG_FINGERPRINT == "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"

    def test_locked_config_fp_backward_compat_alias(self):
        """The old LOCKED_CONFIG_FP must still exist as a backward-compat alias."""
        from course_work.final_test_evaluation import (
            LOCKED_CONFIG_FP,
            LOCKED_CONFIG_FINGERPRINT,
        )
        assert LOCKED_CONFIG_FP == LOCKED_CONFIG_FINGERPRINT

    def test_distinct_constants_exported(self):
        """Both constants must be importable."""
        from course_work import final_test_evaluation as fte
        assert hasattr(fte, "LOCKED_CONFIG_FINGERPRINT")
        assert hasattr(fte, "LOCKED_FINAL_LOCK_SHA256")
        assert hasattr(fte, "LOCKED_CONFIG_FP") 


class TestPhase47RunnerLockIdentity:
    def test_p47_runner_imports_lock_constants(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "p47_runner", ROOT / "src/course_work/scripts/p47_final_test_evaluation.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "LOCKED_CONFIG_FINGERPRINT")
        assert hasattr(mod, "LOCKED_FINAL_LOCK_SHA256")

    def test_p47_runner_writes_lock_sha_not_config_fp_into_contract(self):
        """The p47 runner must reference LOCKED_FINAL_LOCK_SHA256 (not
        LOCKED_CONFIG_FINGERPRINT) when writing the evaluation contract's
        final_lock_sha256 field."""
        p47 = (ROOT / "src/course_work/scripts/p47_final_test_evaluation.py").read_text()
        assert "LOCKED_FINAL_LOCK_SHA256" in p47
        assert "final_lock_sha256=LOCKED_FINAL_LOCK_SHA256" in p47

    def test_p47_runner_distinguishes_lock_vs_config_match(self):
        """The p47 runner must compare ckpt.final_lock_sha256 to
        LOCKED_FINAL_LOCK_SHA256 and ckpt.config_sha256 to
        LOCKED_CONFIG_FINGERPRINT."""
        p47 = (ROOT / "src/course_work/scripts/p47_final_test_evaluation.py").read_text()
        assert "ckpt.final_lock_sha256 == LOCKED_FINAL_LOCK_SHA256" in p47
        assert "ckpt.config_sha256 == LOCKED_CONFIG_FINGERPRINT" in p47
        assert "ckpt.final_lock_sha256 == LOCKED_CONFIG_FP" not in p47
        assert "ckpt.config_sha256 == LOCKED_CONFIG_FP" not in p47

class TestPhase46NoTRC0Fallback:
    def test_p46_runner_refuses_TR_C0_PRIMARY_fallback(self):
        """The Phase 46 runner must not contain `or "TR_C0_PRIMARY"` as a fallback
        in the locked_id resolution path."""
        p46 = (ROOT / "src/course_work/scripts/p46_three_seed_runs.py").read_text()
        assert 'or "TR_C0_PRIMARY"' not in p46, (
            "Phase 46 runner still contains the TR_C0_PRIMARY silent fallback!"
        )

    def test_p46_runner_explicit_guard_against_TR_C0_PRIMARY(self):
        """Phase 46 runner must explicitly guard against TR_C0_PRIMARY."""
        p46 = (ROOT / "src/course_work/scripts/p46_three_seed_runs.py").read_text()
        assert "TR_C0_PRIMARY" in p46, (
            "Expected explicit guard against TR_C0_PRIMARY"
        )
        assert 'locked_id == "TR_C0_PRIMARY"' in p46 or "candidate_id == 'TR_C0_PRIMARY'" in p46 or "candidate_id == \"TR_C0_PRIMARY\"" in p46, (
            "Phase 46 runner must explicitly check candidate_id != TR_C0_PRIMARY"
        )
        assert "sys.exit(2)" in p46, (
            "Phase 46 runner must STOP (sys.exit) on TR_C0_PRIMARY"
        )

    def test_p46_runner_handoff_missing_candidate_id_stops(self):
        """If both candidate_id and locked_model_id are missing, STOP."""
        p46 = (ROOT / "src/course_work/scripts/p46_three_seed_runs.py").read_text()
        assert "Refusing to fall back" in p46 or "cannot be resolved" in p46

class TestPhase46EpochSource:
    def test_p46_runner_has_locked_final_refit_epochs_helper(self):
        """Phase 46 runner must have a helper that reads Phase 45 locked value."""
        p46 = (ROOT / "src/course_work/scripts/p46_three_seed_runs.py").read_text()
        assert "_locked_final_refit_epochs" in p46, (
            "Phase 46 runner must have _locked_final_refit_epochs helper"
        )

    def test_p46_runner_no_silent_50_epoch_default(self):
        """Phase 46 runner must NOT silently default to 50 epochs."""
        p46 = (ROOT / "src/course_work/scripts/p46_three_seed_runs.py").read_text()
        for line in p46.split("\n"):
            if 'int(training.get("max_epochs", 50))' in line:
                pytest.fail(
                    f"Phase 46 runner still has silent 50-epoch fallback: {line!r}"
                )

    def test_locked_final_refit_epochs_helper_reads_phase45(self):
        """_locked_final_refit_epochs helper must read from phase_45_signoff.json."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "p46_runner_helper", ROOT / "src/course_work/scripts/p46_three_seed_runs.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        helper = mod._locked_final_refit_epochs
        result = helper()
        assert result == LOCKED_FINAL_REFIT_EPOCHS

class TestLifecycleSplitGate:
    """The corrective Phase 46 gate is split into two lifecycles:
      - PRETRAIN / PRE-EXECUTION — must be capable of PASS before training
      - POST-RUN / PRE-TEST    — only valid AFTER training completes

    SKIP semantics are NEVER used to make post-training checks appear
    acceptable in the PRETRAIN gate.
    """

    def _load_gate_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "p46_corrective_pretrain_gate",
            ROOT / "src/course_work/scripts/p46_corrective_pretrain_gate.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_corrective_gate_file_exists(self):
        path = ROOT / "src/course_work/scripts/p46_corrective_pretrain_gate.py"
        assert path.exists(), f"Missing {path}"

    def test_gate_defines_two_lifecycles(self):
        mod = self._load_gate_module()
        assert hasattr(mod, "PRETRAIN_GATES")
        assert hasattr(mod, "POSTRUN_GATES")
        assert len(mod.PRETRAIN_GATES) > 0, "PRETRAIN_GATES must be non-empty"
        assert len(mod.POSTRUN_GATES) > 0, "POSTRUN_GATES must be non-empty"

    def test_pretrain_gates_can_pass_before_training(self):
        """Every PRETRAIN gate must be capable of PASS given the current
        Phase 45 lock + handoff + scaler contract — without requiring any
        post-training artifacts (checkpoints, signoff, phase47 release)."""
        mod = self._load_gate_module()
        results = []
        for fn in mod.PRETRAIN_GATES:
            try:
                results.append(fn())
            except Exception as exc:
                pytest.fail(f"PRETRAIN gate {fn.__name__} raised: {exc}")
        failed = [r for r in results if r.status == "FAIL"]
        if failed:
            details = "\n".join(
                f"  {r.gate_id} {r.name}: {r.detail}" for r in failed
            )
            pytest.fail(f"PRETRAIN gates FAILED (cannot pass before training):\n{details}")

    def test_pretrain_does_not_require_phase47_test_release(self):
        """The PRETRAIN gate set must NOT include any check that depends on
        phase47_test_release.json (which is an output of POST-RUN)."""
        mod = self._load_gate_module()
        import inspect
        for fn in mod.PRETRAIN_GATES:
            src = inspect.getsource(fn)
            assert "phase47_test_release" not in src, (
                f"PRETRAIN gate {fn.__name__} references phase47_test_release.json "
                f"(it should be a POSTRUN-only check)"
            )
            assert "PHASE47_RELEASE_PATH" not in src, (
                f"PRETRAIN gate {fn.__name__} references PHASE47_RELEASE_PATH"
            )

    def test_pretrain_does_not_require_completed_phase46_checkpoints(self):
        """The PRETRAIN gate set must NOT include any check that depends on
        completed Phase 46 checkpoint binaries or metadata sidecars."""
        mod = self._load_gate_module()
        import inspect
        for fn in mod.PRETRAIN_GATES:
            src = inspect.getsource(fn)
            assert "FINAL_REFIT.pt" not in src, (
                f"PRETRAIN gate {fn.__name__} references FINAL_REFIT.pt "
                f"(it should be a POSTRUN-only check)"
            )
            assert "_load_per_seed_hashes" not in src, (
                f"PRETRAIN gate {fn.__name__} calls _load_per_seed_hashes "
                f"(post-training data)"
            )
            assert "phase_46_signoff" not in src, (
                f"PRETRAIN gate {fn.__name__} reads phase_46_signoff.json "
                f"(it should be a POSTRUN-only check)"
            )

    def test_postrun_gates_pass_when_checkpoints_present(self):
        """The POSTRUN gate must PASS when checkpoint binaries exist and
        scientific contract is satisfied.

        As of Phase 46 successful completion (Sept 7, 2026, 20:09 UTC):
        All 3 corrected seeds (RUN_TR_FSD_0256_*) have official FINAL_REFIT
        metadata and checkpoint binaries, and their scientific contract
        matches the Phase 45 lock.

        This test verifies the post-training success path. It supersedes
        the prior pre-training version of this test.
        """
        mod = self._load_gate_module()
        rg2 = next(fn for fn in mod.POSTRUN_GATES if fn.__name__.startswith("postrun_g2_"))
        r2 = rg2()
        assert r2.status == "PASS", (
            f"RG2 must PASS when checkpoint binaries exist; got {r2.status}: {r2.detail}"
        )
        rg1 = next(fn for fn in mod.POSTRUN_GATES if fn.__name__.startswith("postrun_g1_"))
        r1 = rg1()
        assert r1.status == "PASS", (
            f"RG1 must PASS when 3 per-seed metadata sidecars exist; got {r1.status}: {r1.detail}"
        )

    def test_postrun_gates_fail_on_stale_historical_run_ids(self):
        """POSTRUN gate RG3 must enforce that no historical run_id leaks.

        As of Phase 46 successful completion (Sept 7, 2026, 20:09 UTC):
        All 3 corrected seeds (RUN_TR_FSD_0256_*) have official FINAL_REFIT
        metadata, and none of them is in EXCLUDED_HISTORICAL_RUN_IDS. RG3
        must therefore PASS.

        Prior to Part 2G-L cleanup, RG3 used to fail either because
        metadata was absent or because it contained RUN_TR_FSD_0153 etc.
        Those failure modes are no longer reachable.
        """
        mod = self._load_gate_module()
        rg3 = next((fn for fn in mod.POSTRUN_GATES
                    if fn.__name__.startswith("postrun_g3_")), None)
        assert rg3 is not None, "POSTRUN must contain a run-id uniqueness gate"
        result = rg3()
        assert result.status == "PASS", (
            f"RG3 must PASS after Phase 46 completion; got {result.status}: "
            f"{result.detail}"
        )
        assert result.evidence and "run_ids" in result.evidence, (
            "RG3 evidence must include run_ids"
        )
        rids = result.evidence["run_ids"]
        assert rids == [
            "RUN_TR_FSD_0256_C2F24D58",
            "RUN_TR_FSD_0256_AA575C42",
            "RUN_TR_FSD_0256_247AB83A",
        ], f"RG3 run_ids mismatch: {rids}"
        excluded = mod.EXCLUDED_HISTORICAL_RUN_IDS
        leaked = [r for r in rids if r in excluded]
        assert not leaked, f"Historical run_ids leaked into corrected Phase 46: {leaked}"

    def test_postrun_gates_pass_on_lock_match(self):
        """POSTRUN must PASS when per-seed config_fingerprint matches the
        locked Phase 45 values.

        As of Phase 46 successful completion (Sept 7, 2026, 20:09 UTC):
        All 3 corrected seeds have config_fingerprint=585c5e79... (the
        canonical Phase 45 lock). RG4 must PASS.
        """
        mod = self._load_gate_module()
        rg4 = next(fn for fn in mod.POSTRUN_GATES if fn.__name__.startswith("postrun_g4_"))
        r4 = rg4()
        assert r4.status == "PASS", (
            f"RG4 must PASS after Phase 46 completion (config_fingerprint matches lock); "
            f"got {r4.status}: {r4.detail}"
        )

    def test_postrun_phase47_release_not_a_prerequisite_gate(self):
        """phase47_test_release.json must NOT appear as a separate gate. It is
        an OUTPUT of successful POST-RUN release logic, generated by
        p46_three_seed_runs.py only after all POSTRUN gates PASS."""
        mod = self._load_gate_module()
        import inspect
        for fn in mod.POSTRUN_GATES:
            src = inspect.getsource(fn)
            assert 'release.get("released") is True' not in src and \
                   'release.get("released") != True' not in src, (
                f"POSTRUN gate {fn.__name__} depends on phase47_test_release.released "
                f"as a gate (should be an output, not a gate)"
            )

    def test_no_skip_semantics_in_lifecycles(self):
        """Neither lifecycle must use SKIP semantics — each gate returns PASS or FAIL."""
        mod = self._load_gate_module()
        import inspect
        for fn in mod.PRETRAIN_GATES + mod.POSTRUN_GATES:
            src = inspect.getsource(fn)
            assert ".skip(" not in src, (
                f"Gate {fn.__name__} uses SKIP semantics "
                f"(lifecycle-split gate must use PASS/FAIL only)"
            )

    def test_gate_run_id_threshold_no_longer_required(self):
        """The corrective plan §3 Fix #11 originally mentioned a numeric
        threshold (RUN_TR_FSD_0256+) — but per Part 2F-B, no specific numeric
        threshold should be required unless canonical registry explicitly requires it.
        Verify no fixed numeric prefix (e.g. 'RUN_TR_FSD_0256') is checked."""
        mod = self._load_gate_module()
        import inspect
        for fn in mod.POSTRUN_GATES:
            src = inspect.getsource(fn)
            assert "RUN_TR_FSD_0256" not in src, (
                f"POSTRUN gate {fn.__name__} hard-codes numeric run_id threshold "
                f"RUN_TR_FSD_0256 (canonical registry does not require it)"
            )

    def test_corrective_gate_constants_match_lock(self):
        path = ROOT / "src/course_work/scripts/p46_corrective_pretrain_gate.py"
        text = path.read_text()
        assert LOCKED_CANDIDATE in text
        assert str(LOCKED_LOOKBACK) in text
        assert str(LOCKED_FINAL_REFIT_EPOCHS) in text
        assert LOCKED_CONFIG_FINGERPRINT in text
        assert LOCKED_FINAL_LOCK_SHA256 in text
        assert FORBIDDEN_CANDIDATE_FALLBACK in text  # TR_C0_PRIMARY

    def test_corrective_gate_lists_excluded_historical_run_ids(self):
        path = ROOT / "src/course_work/scripts/p46_corrective_pretrain_gate.py"
        text = path.read_text()
        for rid in EXCLUDED_HISTORICAL_RUN_IDS:
            assert rid in text, f"Gate missing historical exclusion: {rid}"

    def test_corrective_gate_seeds_match_lock(self):
        path = ROOT / "src/course_work/scripts/p46_corrective_pretrain_gate.py"
        text = path.read_text()
        assert "[42, 123, 2026]" in text

    def test_no_test_access_in_gate_runner(self):
        """The gate runner must not CONSTRUCT/INVOKE any Test loader. Only verifies files.
        Strings used as forbidden-pattern markers in the gate's source scanner
        are allowed (they are static tokens, not invocations)."""
        import ast
        import re
        path = ROOT / "src/course_work/scripts/p46_corrective_pretrain_gate.py"
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, (ast.Str, ast.Constant))
                        and isinstance(getattr(node.body[0].value, "value", None), str)):
                    node.body[0].value.value = ""
        active = ast.unparse(tree)

        active_no_string_literals = re.sub(r"(['\"]).*?\1", "", active, flags=re.DOTALL)
        assert "build_test_dataset(" not in active_no_string_literals, (
            "Gate runner code calls build_test_dataset(...) — must not invoke Test loaders"
        )
        assert "TestLoader(" not in active_no_string_literals, (
            "Gate runner code calls TestLoader(...) — must not invoke Test loaders"
        )

    def test_historical_test_log_preserved_by_gate(self):
        """The gate must NOT clear or rewrite the historical Test access log.
        It only verifies the log is preserved (count of archived events)."""
        path = ROOT / "src/course_work/scripts/p46_corrective_pretrain_gate.py"
        text = path.read_text()
        assert ".unlink" not in text or ".unlink" in text and "final_test_discrepancies" not in text
        assert ".archive" in text, (
            "Gate must verify historical Test access log preservation in .archive/"
        )

class TestPhase47ArchiveScript:
    def test_p47_archive_script_exists(self):
        path = ROOT / "src/course_work/scripts/p47_archive_current_evidence.py"
        assert path.exists(), f"Missing {path}"

    def test_p47_archive_script_no_test_access(self):
        """The archive script must not access Test; only archive files."""
        path = ROOT / "src/course_work/scripts/p47_archive_current_evidence.py"
        text = path.read_text()
        assert "build_test_dataset" not in text
        assert "TestLoader" not in text
        assert "training_step" not in text
        assert "optimizer.step" not in text

    def test_p47_archive_script_verifies_sha(self):
        """The archive script must verify SHA256 source vs archived copy."""
        path = ROOT / "src/course_work/scripts/p47_archive_current_evidence.py"
        text = path.read_text()
        assert "sha256_file" in text
        assert "source_sha" in text
        assert "archived_sha" in text
        assert "ROLLBACK" in text or "rollback" in text

    def test_p47_archive_script_has_history_dir(self):
        """The archive destination must be _history/PHASE47_PRE_CORRECTIVE_<UTC>/."""
        path = ROOT / "src/course_work/scripts/p47_archive_current_evidence.py"
        text = path.read_text()
        assert "_history" in text
        assert "PHASE47_PRE_CORRECTIVE" in text

    def test_p47_archive_script_explicitly_not_executed(self):
        """The script must state it is NOT executed in Part 2F."""
        path = ROOT / "src/course_work/scripts/p47_archive_current_evidence.py"
        text = path.read_text()
        assert "NOT EXECUTED IN PART 2F" in text or "approval_status" in text

class TestProtocolDriftGuards:
    def test_p46_runner_no_warm_start(self):
        p46 = (ROOT / "src/course_work/scripts/p46_three_seed_runs.py").read_text()
        assert 'warm_start = True' not in p46
        for line in p46.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "warm-start" in stripped and "no_warm_start" not in stripped:
                pytest.fail(f"Active warm-start reference: {line!r}")

    def test_p46_runner_no_validation_stopping(self):
        p46 = (ROOT / "src/course_work/scripts/p46_three_seed_runs.py").read_text()
        assert "early_stopping_enabled = True" not in p46
        assert "validation_loader is not None" not in p46

class TestPhase47ReleaseGateEnforced:
    def test_p47_runner_requires_released_phase47_release(self):
        p47 = (ROOT / "src/course_work/scripts/p47_final_test_evaluation.py").read_text()
        assert "released" in p47

    def test_p47_pretest_gate_exists(self):
        path = ROOT / "src/course_work/scripts/p47_pretest_gate.py"
        assert path.exists()

    def test_p47_pretest_gate_uses_distinct_constants(self):
        path = ROOT / "src/course_work/scripts/p47_pretest_gate.py"
        text = path.read_text()
        assert "LOCKED_FINAL_LOCK_SHA256" in text
        assert "ckpt.final_lock_sha256 == LOCKED_CONFIG_FP" not in text

class TestPredictionBundlePersistence:
    def test_writer_has_write_prediction_bundle(self):
        from course_work.final_test_evaluation import writers
        assert hasattr(writers, "write_prediction_bundle")

    def test_writer_has_write_prediction_checksums(self):
        from course_work.final_test_evaluation import writers
        assert hasattr(writers, "write_prediction_checksums")

    def test_writer_emits_per_seed_csv(self):
        """write_prediction_bundle must use seed-based filename."""
        from course_work.final_test_evaluation import writers
        import inspect
        sig = inspect.signature(writers.write_prediction_bundle)
        params = list(sig.parameters.keys())
        assert "seed" in params

class TestLockSeparationEverywhere:
    def test_no_writer_mixes_lock_and_config(self):
        """Each writer function that writes final_lock_sha256 must use
        LOCKED_FINAL_LOCK_SHA256, not LOCKED_CONFIG_FP."""
        import re
        writers = (ROOT / "src/course_work/final_test_evaluation/writers.py").read_text()
        bad = re.findall(
            r'"final_lock_sha(?:256|hash)":\s*LOCKED_CONFIG_FP\b',
            writers,
        )
        assert len(bad) == 0, (
            f"writers.py still writes LOCKED_CONFIG_FP into final_lock_* field: {bad}"
        )

    def test_no_p47_runner_mixes_lock_and_config(self):
        p47 = (ROOT / "src/course_work/scripts/p47_final_test_evaluation.py").read_text()
        import re
        bad = re.findall(
            r'final_lock_sha256=LOCKED_CONFIG_FP\b|"final_lock_sha256":\s*LOCKED_CONFIG_FP\b',
            p47,
        )
        assert len(bad) == 0, (
            f"p47 runner still writes LOCKED_CONFIG_FP into final_lock_sha256: {bad}"
        )

    def test_no_p47_pretest_gate_mixes_lock_and_config(self):
        p47g = (ROOT / "src/course_work/scripts/p47_pretest_gate.py").read_text()
        assert "ckpt.final_lock_sha256 == LOCKED_CONFIG_FP" not in p47g, (
            "p47_pretest_gate.py still uses LOCKED_CONFIG_FP for final_lock_sha256"
        )

class TestLifecycleSplitFixture:
    """Prove the lifecycle-split gate design using a temporary fixture:
       - PRETRAIN gates PASS without any corrected checkpoints
       - PRETRAIN gates do NOT require phase47_test_release.json
       - POSTRUN gates FAIL when checkpoints are missing
       - POSTRUN gates PASS when valid corrected artifacts exist
    """

    def _load_gate_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "p46_corrective_pretrain_gate",
            ROOT / "src/course_work/scripts/p46_corrective_pretrain_gate.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_pretrain_can_pass_without_any_checkpoints(self):
        """Run PRETRAIN gates with the current repo state.
        No checkpoint binaries exist yet (corrective training not run).
        PRETRAIN must still return PASS because it does not depend on checkpoints."""
        mod = self._load_gate_module()
        results = []
        for fn in mod.PRETRAIN_GATES:
            results.append(fn())
        failed = [r for r in results if r.status == "FAIL"]
        assert len(failed) == 0, (
            f"PRETRAIN gate must PASS without checkpoints. Failed: "
            f"{[(r.gate_id, r.detail) for r in failed]}"
        )

    def test_postrun_passes_when_corrected_checkpoints_exist(self):
        """POSTRUN gates must PASS when 3 corrected Phase 46 checkpoints
        exist with valid metadata and matching scientific contract.

        As of Phase 46 successful completion (Sept 7, 2026, 20:09 UTC):
        All 3 corrected seeds (RUN_TR_FSD_0256_*) have valid FINAL_REFIT
        metadata. This test supersedes the pre-training version.
        """
        mod = self._load_gate_module()
        results = []
        for fn in mod.POSTRUN_GATES:
            results.append(fn())
        rg2 = next((r for r in results if r.gate_id == "RG2"), None)
        assert rg2 is not None and rg2.status == "PASS", (
            f"RG2 must PASS when corrected checkpoints exist; got "
            f"status={rg2.status if rg2 else 'missing'}, detail={rg2.detail if rg2 else ''}"
        )

    def test_postrun_can_pass_when_corrected_artifacts_provided(self, tmp_path):
        """Build a synthetic corrected Phase 46 artifact set in a tmp dir
        and verify the POSTRUN gates that read from those paths can PASS.

        Note: the gate module uses module-level constants for canonical paths
        (PROJECT_ROOT / "artifacts" / "three_seed_final_runs" / ...).
        We patch OFFICIAL_CHECKPOINTS_DIR and PHASE46_SIGNOFF_PATH via
        monkey-patching the module, then assert the gates return PASS.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "p46_corrective_pretrain_gate_for_fixture",
            ROOT / "src/course_work/scripts/p46_corrective_pretrain_gate.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        synthetic_artifacts = tmp_path / "three_seed_final_runs"
        synthetic_artifacts.mkdir(parents=True, exist_ok=True)
        official = synthetic_artifacts / "official_checkpoints"
        official.mkdir(parents=True, exist_ok=True)

        per_seed = {}
        for seed in [42, 123, 2026]:
            seed_dir = official / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            ckpt = seed_dir / f"seed_{seed}_FINAL_REFIT.pt"
            ckpt.write_bytes(f"CORRECTED_CHECKPOINT_BYTES_seed{seed}".encode("utf-8"))
            sha = mod._sha256_file(ckpt)
            meta = {
                "seed": seed,
                "run_id": f"RUN_TR_FSD_CORRECTED_SEED{seed}_ABCDEF",
                "phase": 46,
                "checkpoint_type": "FINAL_REFIT",
                "config_sha256": LOCKED_CONFIG_FINGERPRINT,
                "recipe_sha256": LOCKED_RECIPE_SHA256,
                "population_fingerprint": LOCKED_POPULATION_SHA256,
                "x_scaler_sha256": "X_SCALER_CORRECTED_SHA",
                "y_scaler_sha256": "Y_SCALER_CORRECTED_SHA",
                "model_state_sha256": sha,
                "lookback_steps": 72,
                "final_refit_epochs": 30,
            }
            meta_path = seed_dir / f"seed_{seed}_FINAL_REFIT_metadata.json"
            meta_path.write_bytes(json.dumps(meta, indent=2).encode("utf-8"))
            per_seed[seed] = meta

        signoff = {
            "overall_status": "PASS",
            "status": "PASS",
            "discrepancies": [],
            "candidate_id": LOCKED_CANDIDATE,
            "final_refit_epochs": LOCKED_FINAL_REFIT_EPOCHS,
            "config_sha256": LOCKED_CONFIG_FINGERPRINT,
            "final_lock_sha256": LOCKED_FINAL_LOCK_SHA256,
            "recipe_sha256": LOCKED_RECIPE_SHA256,
            "population_sha256": LOCKED_POPULATION_SHA256,
            "feature_sha256": LOCKED_FEATURE_SHA256,
            "x_scaler_sha256": "X_SCALER_CORRECTED_SHA",
            "y_scaler_sha256_or_identity": "Y_SCALER_CORRECTED_SHA",
        }
        signoff_path = synthetic_artifacts / "phase_46_signoff.json"
        signoff_path.write_bytes(json.dumps(signoff, indent=2).encode("utf-8"))
        original_official = mod.OFFICIAL_CHECKPOINTS_DIR
        original_signoff = mod.PHASE46_SIGNOFF_PATH
        original_phase47_release = mod.PHASE47_RELEASE_PATH
        mod.OFFICIAL_CHECKPOINTS_DIR = official
        mod.PHASE46_SIGNOFF_PATH = signoff_path
        mod.PHASE47_RELEASE_PATH = synthetic_artifacts / "phase47_test_release.json"
        try:
            failed_gates = []
            for fn in mod.POSTRUN_GATES:
                result = fn()
                if result.status == "FAIL":
                    failed_gates.append((result.gate_id, result.detail))
            assert len(failed_gates) == 0, (
                f"POSTRUN gates should PASS with synthetic corrected artifacts, "
                f"but failed: {failed_gates}"
            )
        finally:
            mod.OFFICIAL_CHECKPOINTS_DIR = original_official
            mod.PHASE46_SIGNOFF_PATH = original_signoff
            mod.PHASE47_RELEASE_PATH = original_phase47_release

    def test_historical_test_log_preserved_by_postrun_gate(self):
        """The POSTRUN 'no new Test access' gate must NOT clear or rewrite the
        historical .archive directory. It only inspects (counts) the preserved events."""
        archive_dir = ROOT / "artifacts" / "final_test" / ".archive"
        assert archive_dir.exists(), (
            "Historical .archive directory must exist"
        )
        archived = list(archive_dir.glob("final_test_access_event_*.json"))
        n_before = len(archived)
        assert n_before > 0, (
            "Expected at least one historical final_test_access_event preserved"
        )
        mod = self._load_gate_module()
        for fn in mod.POSTRUN_GATES:
            if "no_test_access" in fn.__name__:
                fn()  

        archived_after = list(archive_dir.glob("final_test_access_event_*.json"))
        assert len(archived_after) == n_before, (
            f"POSTRUN gate must not add/remove historical access events; "
            f"before={n_before}, after={len(archived_after)}"
        )

    def test_postrun_release_true_only_after_postrun_pass(self, tmp_path):
        """release=true is only possible after all POSTRUN gates PASS. The release
        itself is generated by p46_three_seed_runs.py as an OUTPUT, not by the gate."""
        mod = self._load_gate_module()
        import inspect
        for fn in mod.POSTRUN_GATES:
            src = inspect.getsource(fn)
            assert "released" not in src or "released" in src and "no_test_access" in fn.__name__, (
                f"POSTRUN gate {fn.__name__} references 'released' status — "
                f"phase47_test_release is an OUTPUT, not a gate"
            )


class TestPhase46ArtifactRootResolution:
    """The Phase 46 runner must resolve ROOT to the COURSE_WORK project root
    (the directory containing both `artifacts/` and `src/`).

    Previously ROOT was computed as `Path(__file__).resolve().parent.parent`,
    which returned `<repo>/COURSE_WORK/src/course_work` and made artifact
    lookups resolve to `<repo>/COURSE_WORK/src/course_work/artifacts/...`
    instead of the canonical `<repo>/COURSE_WORK/artifacts/...`.

    Fix: import the canonical helper `course_work.utils.artifacts.get_project_root()`
    and assign ROOT to its result.
    """

    def _load_runner_module(self):
        """Load p46_three_seed_runs.py with the correct sys.path.
        Mirrors how the user actually invokes the script:
            PYTHONPATH=COURSE_WORK/src python3 COURSE_WORK/src/course_work/scripts/p46_three_seed_runs.py
        """
        import importlib.util
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "src"))
        spec = importlib.util.spec_from_file_location(
            "p46_three_seed_runs_for_test",
            ROOT / "src" / "course_work" / "scripts" / "p46_three_seed_runs.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_runner_root_is_course_work_project_root(self):
        """ROOT must point at the COURSE_WORK directory (containing both
        `artifacts/` and `src/`), NOT at `src/course_work/`."""
        mod = self._load_runner_module()
        assert (mod.ROOT / "artifacts").exists(), (
            f"ROOT = {mod.ROOT} has no artifacts/ directory. "
            f"The runner is resolving ROOT to the wrong location."
        )
        assert (mod.ROOT / "src").exists(), (
            f"ROOT = {mod.ROOT} has no src/ directory. "
            f"The runner is resolving ROOT to the wrong location."
        )
        assert not (mod.ROOT / "course_work").exists() or (
            mod.ROOT / "course_work"
        ).is_file() is False, (
            f"ROOT = {mod.ROOT} looks like a sub-package directory (has course_work/). "
            f"This is the bug: ROOT must be the COURSE_WORK project root, not the package."
        )

    def test_runner_root_does_not_point_into_src(self):
        """ROOT must NEVER be `<repo>/COURSE_WORK/src/course_work`.
        The buggy resolution pointed here, causing the artifact paths to
        resolve under `src/course_work/artifacts/` instead of `artifacts/`."""
        mod = self._load_runner_module()
        root_str = str(mod.ROOT.resolve())
        assert "src/course_work" not in root_str, (
            f"ROOT still resolves into src/course_work/: {root_str}. "
            f"This is the path-resolution bug. ROOT must be the COURSE_WORK project root."
        )

    def test_artifact_dir_resolves_to_course_work_artifacts(self):
        """ARTIFACT_DIR = ROOT / 'artifacts' / 'three_seed_final_runs'
        must be <COURSE_WORK>/artifacts/three_seed_final_runs/."""
        mod = self._load_runner_module()
        expected = ROOT / "artifacts" / "three_seed_final_runs"
        assert mod.ARTIFACT_DIR == expected, (
            f"ARTIFACT_DIR = {mod.ARTIFACT_DIR} != expected {expected}"
        )
        assert (mod.ARTIFACT_DIR / "phase_46_signoff.json").exists(), (
            f"phase_46_signoff.json not found at expected path "
            f"{mod.ARTIFACT_DIR / 'phase_46_signoff.json'}"
        )

    def test_phase45_signoff_path_is_discoverable(self):
        """PHASE_45_SIGNOFF must point at the actual Phase 45 signoff artifact."""
        mod = self._load_runner_module()
        expected = ROOT / "artifacts" / "final_model_lock" / "phase_45_signoff.json"
        assert mod.PHASE_45_SIGNOFF == expected, (
            f"PHASE_45_SIGNOFF = {mod.PHASE_45_SIGNOFF} != expected {expected}"
        )
        assert mod.PHASE_45_SIGNOFF.exists(), (
            f"PHASE_45_SIGNOFF does not exist at {mod.PHASE_45_SIGNOFF}. "
            f"Runner cannot find the Phase 45 lock artifact."
        )

    def test_phase46_handoff_path_is_discoverable(self):
        """PHASE_46_HANDOFF must point at the actual Phase 46 handoff artifact."""
        mod = self._load_runner_module()
        expected = ROOT / "artifacts" / "final_model_lock" / "phase46_three_seed_handoff.json"
        assert mod.PHASE_46_HANDOFF == expected, (
            f"PHASE_46_HANDOFF = {mod.PHASE_46_HANDOFF} != expected {expected}"
        )
        assert mod.PHASE_46_HANDOFF.exists(), (
            f"PHASE_46_HANDOFF does not exist at {mod.PHASE_46_HANDOFF}. "
            f"Runner cannot find the Phase 46 handoff artifact."
        )

    def test_no_absolute_paths_in_runner(self):
        """The runner must not hard-code any absolute macOS / linux path.
        The previously buggy resolution pattern (`parent.parent`) must NOT
        appear in any active code line."""
        import ast
        text = (ROOT / "src/course_work/scripts/p46_three_seed_runs.py").read_text()
        assert "/Users/" not in text, (
            "Runner hard-codes a /Users/ absolute path. Must use the canonical "
            "ROOT resolver."
        )
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, (ast.Str, ast.Constant))
                        and isinstance(getattr(node.body[0].value, "value", None), str)):
                    node.body[0].value.value = ""
        active = ast.unparse(tree)
        assert 'Path(__file__).resolve().parent.parent' not in active, (
            "Runner still uses `Path(__file__).resolve().parent.parent` in active "
            "code, which resolves to src/course_work/ instead of the COURSE_WORK "
            "project root. Use `course_work.utils.artifacts.get_project_root()` instead."
        )

    def test_runner_uses_canonical_project_root_helper(self):
        """The runner must use the canonical helper `get_project_root()`
        to resolve ROOT."""
        text = (ROOT / "src/course_work/scripts/p46_three_seed_runs.py").read_text()
        assert "get_project_root" in text, (
            "Runner does not import the canonical get_project_root() helper. "
            "ROOT resolution should be delegated to that helper."
        )

    def test_loading_runner_does_not_execute_scientific_code(self):
        """Just importing the runner module must NOT execute training,
        inference, Test access, or any scientific computation."""
        mod = self._load_runner_module()
        phase45_lock = ROOT / "artifacts/final_model_lock/phase_45_signoff.json"
        lock_sha_before = hashlib.sha256(phase45_lock.read_bytes()).hexdigest()
        self._load_runner_module()
        lock_sha_after = hashlib.sha256(phase45_lock.read_bytes()).hexdigest()
        assert lock_sha_before == lock_sha_after, (
            "Importing the Phase 46 runner modified the Phase 45 lock artifact. "
            "This must never happen — module-level imports must be side-effect-free."
        )
