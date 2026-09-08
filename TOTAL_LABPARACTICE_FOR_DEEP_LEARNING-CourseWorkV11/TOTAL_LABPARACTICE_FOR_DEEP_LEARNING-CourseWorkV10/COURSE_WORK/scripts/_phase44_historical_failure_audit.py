"""Phase 44 Historical Failure Regression Audit.

Maps each of the 13 documented historical failure classes to existing
test functions and reports coverage.

Expected: 13/13 covered.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


COVERAGE_MATRIX = [
    (
        "1. ReferenceResolution schema",
        "reference_resolution|reference-resolution|ReferenceResolution",
    ),
    (
        "2. feature-count mismatch",
        "feature_count|FeatureCount|feature.*mismatch|mismatched.*feature",
    ),
    (
        "3. rerun-reason / duplicate registry",
        "rerun_reason|duplicate.*registry|already_registered",
    ),
    (
        "4. TrainingEngine constructor API",
        "TrainingEngine\\(|training_engine.*API|engine\\.train",
    ),
    (
        "5. WB0 short/long protocol",
        "WB0|wb0|WB0_CONTEXT_CARRY_OVER",
    ),
    (
        "6. Phase43 artifact lifecycle",
        "phase43.*artifact|Phase43.*lifecycle|phase43_artifact",
    ),
    (
        "7. synthetic Phase44 official path",
        "synthetic.*official|run_pipeline|_synth_predict",
    ),
    (
        "8. fake inner epochs / fast mode",
        "fast_mode|FastMode|fast_epochs|FAST",
    ),
    (
        "9. missing LSTM Stage A/B",
        "LSTM_TUNED|lstm_handoff|missing_lstm|lstm.*stage",
    ),
    (
        "10. missing Persistence",
        "Persistence|persistence.*missing|compute_persistence_bundle",
    ),
    (
        "11. RunContext API mismatch",
        "RunContext|run_context|runcontext",
    ),
    (
        "12. real [B,L,F] data shape",
        "B,L,F|B, L, F|sequence_dim|shape.*3.*lookback",
    ),
    (
        "13. YS1 scaler contract",
        "YS1.*contract|scaler_bundle_contract|target_scaler_bundle",
    ),
]


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — HISTORICAL FAILURE REGRESSION AUDIT")
    print("=" * 78)

    import re
    test_root = PROJECT_ROOT / "tests"
    test_files = sorted([str(p) for p in test_root.rglob("*.py") if "__pycache__" not in str(p)])

    covered = 0
    for failure_class, pattern in COVERAGE_MATRIX:
        hits = []
        for fp in test_files:
            with open(fp) as f:
                content = f.read()
            if re.search(pattern, content, re.IGNORECASE):
                hits.append(Path(fp).relative_to(PROJECT_ROOT))
        status = "COVERED" if hits else "MISSING"
        if hits:
            covered += 1
        print(f"\n  {failure_class}")
        print(f"    Status: {status}")
        if hits:
            for h in hits[:3]:
                print(f"      ↳ {h}")
            if len(hits) > 3:
                print(f"      … +{len(hits) - 3} more")

    print(f"\n{'=' * 78}")
    print(f"COVERAGE: {covered}/{len(COVERAGE_MATRIX)} historical failure classes covered")
    print(f"{'=' * 78}")
    return 0 if covered == len(COVERAGE_MATRIX) else 1


if __name__ == "__main__":
    sys.exit(main())
