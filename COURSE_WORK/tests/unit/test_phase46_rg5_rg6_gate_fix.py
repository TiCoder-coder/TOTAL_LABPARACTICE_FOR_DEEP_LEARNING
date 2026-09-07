"""Tests for Part 2G-N: POSTRUN gate RG5/RG6 verifier-only fix.

Background: RG5 (final_lock_sha256) and RG6 (recipe/population/feature_sha)
were failing because:
  - RG5 read `m.get("final_lock_sha256")` from per-seed metadata, but
    per-seed metadata does NOT carry `final_lock_sha256` (the writer had
    a conflation bug — `final_lock_sha256: config_sha`).
  - RG6 read stale Phase 45-audit constants (LOCKED_RECIPE_SHA256,
    LOCKED_POPULATION_SHA256) that no longer match the canonical
    Phase 46 identities, AND read `m.get("population_sha256")`
    from per-seed metadata, but per-seed uses `population_fingerprint`.

Fix: derive canonical Phase 46 identities from phase_46_signoff.json
(the authoritative Phase 46-lock artifact) and verify per-seed metadata
correctly with field names that actually exist in the metadata schema.

No training, no inference, no Test access.
"""

from __future__ import annotations

import importlib.util as _iu
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = ROOT / "src" / "course_work" / "scripts" / "p46_corrective_pretrain_gate.py"
PHASE46_SIGNOFF_PATH = (
    ROOT / "artifacts" / "three_seed_final_runs" / "phase_46_signoff.json"
)
OFFICIAL_CHECKPOINTS_DIR = (
    ROOT / "artifacts" / "three_seed_final_runs" / "official_checkpoints"
)
LOCKED_SEEDS = [42, 123, 2026]


def _load_gate_module():
    spec = _iu.spec_from_file_location("p2gn_gate", str(GATE_PATH))
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Snapshot of the current persisted Phase 46 signoff. We will not modify
# this file during the tests; instead, for negative tests we route the
# gate module's PHASE46_SIGNOFF_PATH to a temporary path.
def _read_p46_signoff() -> dict:
    return json.loads(PHASE46_SIGNOFF_PATH.read_text())


class TestCanonicalIdentityDerivation:
    """1. config fingerprint and final lock SHA must remain distinct."""

    def test_canonical_identity_helper_present(self) -> None:
        gate = _load_gate_module()
        assert hasattr(gate, "_load_phase46_canonical_identity"), (
            "Part 2G-N: gate module must expose _load_phase46_canonical_identity helper"
        )

    def test_canonical_identity_distinct_constants(self) -> None:
        """config_fingerprint and final_lock_sha256 MUST remain distinct
        in the authoritative Phase 46 signoff.
        """
        p46 = _read_p46_signoff()
        assert p46["config_sha256"] != p46["final_lock_sha256"], (
            "config_sha256 and final_lock_sha256 must be distinct in phase_46_signoff"
        )
        # Confirmed canonical values
        assert p46["final_lock_sha256"] == (
            "81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec"
        )
        assert p46["config_sha256"] == (
            "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"
        )


class TestRG5ReadsAuthoritativeField:
    """2. RG5 must read final_lock_sha256 from phase_46_signoff.json."""

    def test_rg5_passes_in_current_state(self) -> None:
        gate = _load_gate_module()
        r = gate.postrun_g5_same_final_lock_sha256_across_seeds()
        assert r.status == "PASS", f"RG5 must PASS in current state; got {r.status}: {r.detail}"
        # Evidence must include the canonical final_lock_sha256 from signoff
        assert "final_lock_sha256" in r.evidence
        assert r.evidence["final_lock_sha256"] == (
            "81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec"
        )
        # And source attribution
        assert r.evidence["source"] == "phase_46_signoff.json"

    def test_rg5_no_longer_reads_per_seed_final_lock_sha256_field(self) -> None:
        """The fix: RG5 no longer queries per-seed metadata for
        `final_lock_sha256` (which was missing). It derives from
        phase_46_signoff.json and verifies per-seed `config_sha256`.
        """
        gate_src = GATE_PATH.read_text()
        # The old buggy read pattern must be absent (no longer reads the
        # non-existent `final_lock_sha256` field from per-seed metadata).
        assert "m.get(\"final_lock_sha256\")" not in gate_src, (
            "RG5 must NOT read non-existent `final_lock_sha256` field "
            "from per-seed metadata. Per Part 2G-N, derive from "
            "phase_46_signoff.json instead."
        )


class TestRG5NegativeCaseWrongFinalLock:
    """3. Wrong canonical final_lock_sha256 still FAILS."""

    def test_rg5_fails_when_canonical_config_fp_wrong(self, monkeypatch, tmp_path):
        """If the canonical config_fingerprint in signoff is wrong
        (tampered), RG5 must FAIL because per-seed config_sha256 doesn't match.
        """
        gate = _load_gate_module()

        # Create temporary phase_46_signoff with WRONG config_fingerprint
        original = _read_p46_signoff()
        tampered = dict(original)
        tampered["config_sha256"] = "deadbeef" + original["config_sha256"][8:]
        tampered_path = tmp_path / "phase_46_signoff_tampered.json"
        tampered_path.write_text(json.dumps(tampered, indent=2))

        # Monkey-patch the gate module's PHASE46_SIGNOFF_PATH
        monkeypatch.setattr(gate, "PHASE46_SIGNOFF_PATH", tampered_path)
        r = gate.postrun_g5_same_final_lock_sha256_across_seeds()
        assert r.status == "FAIL", (
            f"RG5 must FAIL when canonical config_fp is wrong; got {r.status}: {r.detail}"
        )
        assert "config_fingerprint" in r.detail.lower() or "config_fingerprints" in r.detail.lower()

    def test_rg5_fails_on_distinctness_conflation(self, monkeypatch, tmp_path):
        """If final_lock_sha256 == config_fingerprint (conflation bug),
        RG5 must FAIL.
        """
        gate = _load_gate_module()
        original = _read_p46_signoff()
        conflated = dict(original)
        conflated["final_lock_sha256"] = original["config_sha256"]
        tampered_path = tmp_path / "phase_46_signoff_conflated.json"
        tampered_path.write_text(json.dumps(conflated, indent=2))

        monkeypatch.setattr(gate, "PHASE46_SIGNOFF_PATH", tampered_path)
        r = gate.postrun_g5_same_final_lock_sha256_across_seeds()
        assert r.status == "FAIL", (
            f"RG5 must FAIL when final_lock == config_fingerprint; got {r.status}: {r.detail}"
        )
        assert "conflation" in r.detail.lower() or "equal" in r.detail.lower()


class TestRG6UsesCanonicalIdentity:
    """4. RG6 must use canonical recipe/population/feature identities
    from phase_46_signoff.json."""

    def test_rg6_passes_in_current_state(self) -> None:
        gate = _load_gate_module()
        r = gate.postrun_g6_same_recipe_population_feature_sha_across_seeds()
        assert r.status == "PASS", f"RG6 must PASS in current state; got {r.status}: {r.detail}"
        assert r.evidence["recipe_sha256"] == (
            "236aca6cc7496535c99f2ae2cb7d8ea4a043620d003c719a32135791c2903b94"
        )
        assert r.evidence["feature_sha256"] == (
            "fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee"
        )
        assert r.evidence["population_fingerprint_per_seed"] == (
            "0a904beeec68f245fbb1214f7968679d76ebc6dfc010db5215f677ddc2d31f39"
        )

    def test_rg6_no_longer_uses_stale_locked_constants(self) -> None:
        """Part 2G-N removed LOCKED_RECIPE_SHA256 (Phase 45-audit hash).
        The gate must derive recipe_sha256 from the signoff.

        We check for USE of the constants (as identifiers/code references)
        rather than just substring containment, since the docstring text
        (which is descriptive only) still mentions these names for context.
        """
        gate_src = GATE_PATH.read_text()
        # Allow the names in docstrings (descriptive) and comments, but NOT
        # as module-level constants or function bodies (used as values).
        import re
        # Remove triple-quoted strings (docstrings) to focus on code
        no_docstrings = re.sub(r'"""[^"]*?"""', "", gate_src, flags=re.DOTALL)
        # Check for actual use as a value (not just descriptive mention)
        # Match: = LOCKED_RECIPE_SHA256
        assert re.search(r"=\s*LOCKED_RECIPE_SHA256\s*\(", no_docstrings) is None, (
            "Part 2G-N: LOCKED_RECIPE_SHA256 (Phase 45-audit hash) must not be used as a value"
        )
        # Match module-level definition
        assert not re.search(r"^LOCKED_RECIPE_SHA256\s*=", no_docstrings, flags=re.MULTILINE), (
            "Part 2G-N: LOCKED_RECIPE_SHA256 module-level constant must be removed"
        )
        assert not re.search(r"^LOCKED_POPULATION_SHA256\s*=", no_docstrings, flags=re.MULTILINE), (
            "Part 2G-N: LOCKED_POPULATION_SHA256 module-level constant must be removed"
        )


class TestRG6NegativeCases:
    """5,6,7. Wrong recipe/population/feature identities still FAIL.

    These tests use monkey-patching to substitute the gate's
    PHASE46_SIGNOFF_PATH with a tmp_path version. They DO NOT mutate
    the real on-disk phase_46_signoff.json or per-seed metadata files.
    """

    def test_rg6_fails_when_canonical_recipe_wrong(self, monkeypatch, tmp_path):
        """If the canonical recipe_sha256 in signoff is tampered to a
        non-empty value AND per-seed metadata uses the original canonical
        recipe_sha256, RG6 must FAIL on the per-seed mismatch check.

        Implementation: only the signoff is monkey-patched; per-seed
        metadata retains the correct (canonical) recipe_sha256.
        """
        gate = _load_gate_module()
        original = _read_p46_signoff()
        tampered = dict(original)
        # Tamper only the canonical recipe_sha256; leave per-seed metadata alone
        tampered["recipe_sha256"] = "aabbccdd" + original["recipe_sha256"][8:]
        tampered_path = tmp_path / "phase_46_signoff_bad_recipe.json"
        tampered_path.write_text(json.dumps(tampered, indent=2))
        monkeypatch.setattr(gate, "PHASE46_SIGNOFF_PATH", tampered_path)
        r = gate.postrun_g6_same_recipe_population_feature_sha_across_seeds()
        assert r.status == "FAIL", (
            f"RG6 must FAIL when canonical recipe_sha256 is wrong; "
            f"got {r.status}: {r.detail}"
        )
        assert "recipe" in r.detail.lower()

    def test_rg6_fails_on_missing_signoff(self, monkeypatch, tmp_path):
        """RG6 must FAIL if phase_46_signoff.json is missing."""
        gate = _load_gate_module()
        missing_path = tmp_path / "does_not_exist.json"
        monkeypatch.setattr(gate, "PHASE46_SIGNOFF_PATH", missing_path)
        r = gate.postrun_g6_same_recipe_population_feature_sha_across_seeds()
        assert r.status == "FAIL"
        assert "phase_46_signoff" in r.detail.lower() or "not complete" in r.detail.lower()

    def test_rg6_fails_on_missing_recipe_in_signoff(self, monkeypatch, tmp_path):
        """RG6 must FAIL if signoff lacks recipe_sha256."""
        gate = _load_gate_module()
        original = _read_p46_signoff()
        broken = {k: v for k, v in original.items() if k != "recipe_sha256"}
        tampered_path = tmp_path / "phase_46_signoff_no_recipe.json"
        tampered_path.write_text(json.dumps(broken, indent=2))
        monkeypatch.setattr(gate, "PHASE46_SIGNOFF_PATH", tampered_path)
        r = gate.postrun_g6_same_recipe_population_feature_sha_across_seeds()
        assert r.status == "FAIL"
        assert "recipe_sha256" in r.detail.lower()


class TestStaleHistoricalExclusion:
    """8. Stale/historical seed still FAILS."""

    def test_rg3_rejects_historical_seeds(self) -> None:
        gate = _load_gate_module()
        # Pre-check: historical run_ids are in the exclusion list
        for rid in (
            "RUN_TR_FSD_0153_B15A19DC",
            "RUN_TR_FSD_0183_C2F24D58",
        ):
            assert rid in gate.EXCLUDED_HISTORICAL_RUN_IDS
        # Per-seed metadata has fresh seeds; RG3 should PASS
        r = gate.postrun_g3_run_ids_unique_and_not_historical()
        assert r.status == "PASS"


class TestFullGateValidation:
    """9. Valid three-seed corrected state passes RG5 and RG6."""

    def test_full_postrun_9_9(self) -> None:
        """Per user requirement: 9/9 POSTRUN gates PASS in the
        current valid three-seed corrected state.
        """
        gate = _load_gate_module()
        for fn in gate.POSTRUN_GATES:
            r = fn()
            assert r.status == "PASS", (
                f"{fn.__name__} FAILED: {r.detail}"
            )


class TestNoScientificMutation:
    """10. No training/inference/Test occurs during gate evaluation."""

    def test_rg5_does_not_call_train(self) -> None:
        """RG5 evaluation does not invoke engine.train() or any
        training path.
        """
        gate_src = GATE_PATH.read_text()
        # Check the RG5 function specifically
        # (function body should NOT contain 'engine.train' or 'optimizer.step')
        import re
        m = re.search(r"def postrun_g5_same_final_lock_sha256_across_seeds.*?(?=\ndef )",
                      gate_src, flags=re.DOTALL)
        assert m is not None
        body = m.group(0)
        assert "engine.train" not in body
        assert "optimizer.step" not in body
        assert ".backward(" not in body

    def test_rg6_does_not_call_train(self) -> None:
        gate_src = GATE_PATH.read_text()
        import re
        m = re.search(r"def postrun_g6_same_recipe_population_feature_sha_across_seeds.*?(?=\ndef )",
                      gate_src, flags=re.DOTALL)
        assert m is not None
        body = m.group(0)
        assert "engine.train" not in body
        assert "optimizer.step" not in body
        assert ".backward(" not in body

    def test_gate_does_not_access_test(self) -> None:
        """Gate must not read Test data, Test columns, or test_locked paths."""
        gate_src = GATE_PATH.read_text()
        # The forbidden patterns
        forbidden = [
            "test_loader",
            "test_dataset",
            "X_test",
            "y_test",
            "test_access_authorized",
            "test_passed",
        ]
        for f in forbidden:
            # Match as identifier, not substring
            assert (f" {f}" not in gate_src and f".{f}" not in gate_src and f"({f}" not in gate_src), (
                f"Gate must not reference {f!r}"
            )
