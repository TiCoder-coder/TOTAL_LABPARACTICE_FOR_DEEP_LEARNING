"""Phase 53 — preflight acceptance tests.

Verifies:

* Phase52 signoff is PASS or PASS_WITH_WARNING
* phase53_ready = true
* 3 raw dense NPZ files exist
* Raw SHA256 matches the expected SHA256 in phase53_attention_heatmaps_handoff.json
* dtype = float32
* shape = [44, 2, 4, 72, 72] per seed
* Relative position map present
* Case metadata present
* Phase51 shared top5 available
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_phase53_artifacts():
    p52_so_fp = ROOT / "artifacts/attention_extraction/phase_52_signoff.json"
    p53_handoff_fp = ROOT / "artifacts/attention_extraction/phase53_attention_heatmaps_handoff.json"
    rpm_fp = ROOT / "artifacts/attention_extraction/attention_relative_position_map.csv"
    case_meta_fp = ROOT / "artifacts/attention_extraction/attention_case_metadata.csv"
    ws_top20_fp = ROOT / "artifacts/worst_error_analysis/worst_shared_top20.csv"
    p53_so_fp = ROOT / "artifacts/attention_heatmaps/phase_53_signoff.json"
    p53_preflight_fp = ROOT / "artifacts/attention_heatmaps/phase53_preflight_audit.csv"

    return {
        "phase52_signoff": json.loads(p52_so_fp.read_text()) if p52_so_fp.exists() else {},
        "phase53_handoff": json.loads(p53_handoff_fp.read_text()) if p53_handoff_fp.exists() else {},
        "raw_dir": ROOT / "artifacts/attention_extraction/raw",
        "rpm_exists": rpm_fp.exists(),
        "case_meta_exists": case_meta_fp.exists(),
        "ws_top20_exists": ws_top20_fp.exists(),
        "phase53_signoff": json.loads(p53_so_fp.read_text()) if p53_so_fp.exists() else {},
        "phase53_preflight": p53_preflight_fp,
    }


def test_phase53_preflight_phase52_signoff():
    a = _load_phase53_artifacts()
    s = a["phase52_signoff"].get("phase52_status", "")
    assert s in ("PASS", "PASS_WITH_WARNING"), f"Phase52 status {s!r} not PASS/PASS_WITH_WARNING"


def test_phase53_preflight_handoff_ready():
    a = _load_phase53_artifacts()
    assert a["phase53_handoff"].get("ready_for_phase53") is True


def test_phase53_preflight_raw_files_exist():
    a = _load_phase53_artifacts()
    for seed in (42, 123, 2026):
        fp = a["raw_dir"] / f"dense_case_attention_seed{seed}.npz"
        assert fp.is_file(), f"Missing raw NPZ for seed {seed}: {fp}"


def test_phase53_preflight_raw_sha_match():
    a = _load_phase53_artifacts()
    expected = a["phase53_handoff"]["raw_files"]
    for seed in (42, 123, 2026):
        fname = f"dense_case_attention_seed{seed}.npz"
        fp = a["raw_dir"] / fname
        import hashlib
        observed = hashlib.sha256(fp.read_bytes()).hexdigest()
        assert observed == expected[fname], (
            f"SHA drift for {fname}: expected {expected[fname]}, got {observed}"
        )


def test_phase53_preflight_raw_dtype_and_shape():
    a = _load_phase53_artifacts()
    for seed in (42, 123, 2026):
        fp = a["raw_dir"] / f"dense_case_attention_seed{seed}.npz"
        npz = np.load(fp, allow_pickle=False)
        arr = npz["attention"]
        assert str(arr.dtype) == "float32", f"seed {seed}: dtype {arr.dtype} != float32"
        assert list(arr.shape) == [44, 2, 4, 72, 72], (
            f"seed {seed}: shape {arr.shape} != [44, 2, 4, 72, 72]"
        )


def test_phase53_preflight_relative_position_map():
    a = _load_phase53_artifacts()
    assert a["rpm_exists"]


def test_phase53_preflight_case_metadata():
    a = _load_phase53_artifacts()
    assert a["case_meta_exists"]


def test_phase53_preflight_phase51_shared_top5():
    a = _load_phase53_artifacts()
    assert a["ws_top20_exists"]


def test_phase53_signoff_present():
    a = _load_phase53_artifacts()
    so = a["phase53_signoff"]
    assert so.get("phase") == 53
    assert so.get("overall_status") in ("PASS", "PASS_WITH_WARNING")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
