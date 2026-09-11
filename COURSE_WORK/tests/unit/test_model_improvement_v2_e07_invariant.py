"""E07-G1: Multi-candidate invariant tests for E07.

E07 is intentionally a 2-challenger experiment:

  1. TR_C2_ALT_LOOKBACK_SCHED_COSINE
  2. TR_C2_ALT_LOOKBACK_SCHED_REDUCE_ON_PLATEAU

Control E01 is reused read-only and is NOT a training candidate.

These tests verify that:
  - Exactly 2 valid challengers -> PASS
  - 1 challenger -> FAIL
  - 3 challengers -> FAIL
  - 0 challengers -> FAIL
  - wrong candidate IDs -> FAIL
  - scheduler OFF injected as challenger -> FAIL (rejected at preflight level)
  - E06 multi-candidate behavior unchanged (3 LR challengers still required)
"""
from __future__ import annotations

import dataclasses
from copy import deepcopy
from pathlib import Path

import pytest

from course_work.model_improvement_v2.e07_runner import (
    CHALLENGER_CANDIDATE_IDS,
    build_e07_challenger_candidates,
    build_e07_run_context,
    load_e07_config,
    project_root,
)
from course_work.rolling_origin.real_run import assert_context_invariants


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _e07_document():
    return load_e07_config(PROJECT_ROOT)


def _e07_context(document=None):
    if document is None:
        document = _e07_document()
    return build_e07_run_context(PROJECT_ROOT, document)


def _e07_candidates(document=None):
    if document is None:
        document = _e07_document()
    return build_e07_challenger_candidates(document, project_root())


# ----------------------------------------------------------------------
# 1. Exactly 2 valid challengers -> PASS
# ----------------------------------------------------------------------
def test_e07_two_valid_challengers_pass():
    """E07 with the two locked scheduler challengers must pass invariant check."""
    context = _e07_context()
    assert len(context.candidate_specs) == 2
    actual_ids = {candidate.candidate_id for candidate in context.candidate_specs}
    assert actual_ids == {
        "TR_C2_ALT_LOOKBACK_SCHED_COSINE",
        "TR_C2_ALT_LOOKBACK_SCHED_REDUCE_ON_PLATEAU",
    }
    # Must not raise
    assert_context_invariants(context)


def test_e07_candidate_ids_exact():
    """E07 CHALLENGER_CANDIDATE_IDS are the locked 2 scheduler challengers."""
    assert sorted(CHALLENGER_CANDIDATE_IDS) == [
        "TR_C2_ALT_LOOKBACK_SCHED_COSINE",
        "TR_C2_ALT_LOOKBACK_SCHED_REDUCE_ON_PLATEAU",
    ]


# ----------------------------------------------------------------------
# 2. Wrong count challengers -> FAIL
# ----------------------------------------------------------------------
def test_e07_one_challenger_fails():
    """E07 with only 1 candidate must fail."""
    context = _e07_context()
    # Drop one challenger
    context.candidate_specs = context.candidate_specs[:1]
    with pytest.raises(RuntimeError, match="two locked scheduler challengers"):
        assert_context_invariants(context)


def test_e07_three_challengers_fails():
    """E07 with 3 candidates must fail."""
    context = _e07_context()
    # Add a third challenger via dataclasses.replace (CandidateSpec is frozen)
    extra = dataclasses.replace(
        context.candidate_specs[0],
        candidate_id="TR_C2_ALT_LOOKBACK_SCHED_FAKE",
    )
    context.candidate_specs = (*context.candidate_specs, extra)
    with pytest.raises(RuntimeError, match="two locked scheduler challengers"):
        assert_context_invariants(context)


def test_e07_zero_challengers_fails():
    """E07 with 0 candidates must fail."""
    context = _e07_context()
    context.candidate_specs = []
    with pytest.raises(RuntimeError, match="two locked scheduler challengers"):
        assert_context_invariants(context)


# ----------------------------------------------------------------------
# 3. Wrong candidate IDs -> FAIL
# ----------------------------------------------------------------------
def test_e07_wrong_candidate_id_fails():
    """E07 with wrong candidate IDs (e.g. E06 LR candidates) must fail."""
    context = _e07_context()
    # Replace candidate IDs with E06-style IDs (frozen dataclass -> replace)
    context.candidate_specs = (
        dataclasses.replace(context.candidate_specs[0],
                            candidate_id="TR_C2_ALT_LOOKBACK_LR_1E4"),
        dataclasses.replace(context.candidate_specs[1],
                            candidate_id="TR_C2_ALT_LOOKBACK_LR_2E4"),
    )
    with pytest.raises(RuntimeError, match="two locked scheduler challengers"):
        assert_context_invariants(context)


def test_e07_off_as_challenger_id_fails():
    """E07 with OFF-like candidate ID must fail."""
    context = _e07_context()
    # candidate_specs is a tuple; convert to list to allow mutation.
    specs = list(context.candidate_specs)
    specs[0] = dataclasses.replace(
        specs[0],
        candidate_id="TR_C2_ALT_LOOKBACK_SCHED_OFF",
    )
    context.candidate_specs = tuple(specs)
    with pytest.raises(RuntimeError, match="two locked scheduler challengers"):
        assert_context_invariants(context)


# ----------------------------------------------------------------------
# 4. Architecture check: E07 accepts TR_C2_ALT_LOOKBACK_SCHED_*
# ----------------------------------------------------------------------
def test_e07_architecture_accepts_sched_variants():
    """E07 invariant accepts TR_C2_ALT_LOOKBACK_SCHED_* architecture prefix."""
    context = _e07_context()
    # The two locked challengers both use the _SCHED_ suffix.
    for candidate in context.candidate_specs:
        assert candidate.candidate_id.startswith("TR_C2_ALT_LOOKBACK")
    # Must not raise
    assert_context_invariants(context)


def test_e07_architecture_rejects_foreign_candidate_id():
    """E07 invariant rejects candidate IDs that are not TR_C2_ALT_LOOKBACK[-SCHED_*]."""
    context = _e07_context()
    specs = list(context.candidate_specs)
    specs[0] = dataclasses.replace(
        specs[0],
        candidate_id="TR_DIFFERENT",
    )
    context.candidate_specs = tuple(specs)
    # Count is still 2 but the set is wrong -> fails on locked-IDs check.
    with pytest.raises(RuntimeError, match="two locked scheduler challengers"):
        assert_context_invariants(context)


# ----------------------------------------------------------------------
# 5. E06 multi-candidate behavior unchanged
# ----------------------------------------------------------------------
def test_e06_three_candidates_still_pass():
    """E06 invariant must still accept exactly 3 LR challengers."""
    from course_work.model_improvement_v2.e06_runner import (
        CANDIDATE_IDS,
        build_e06_run_context,
        load_e06_config,
    )
    document = load_e06_config(PROJECT_ROOT)
    context = build_e06_run_context(PROJECT_ROOT, document)
    assert len(context.candidate_specs) == 3
    actual_ids = {candidate.candidate_id for candidate in context.candidate_specs}
    assert actual_ids == set(CANDIDATE_IDS.values())
    # Must not raise
    assert_context_invariants(context)


def test_e06_wrong_candidate_count_fails():
    """E06 invariant still rejects wrong counts."""
    from course_work.model_improvement_v2.e06_runner import (
        build_e06_run_context,
        load_e06_config,
    )
    document = load_e06_config(PROJECT_ROOT)
    context = build_e06_run_context(PROJECT_ROOT, document)
    # Drop one candidate
    context.candidate_specs = context.candidate_specs[:2]
    with pytest.raises(RuntimeError, match="three locked LR challengers"):
        assert_context_invariants(context)


# ----------------------------------------------------------------------
# 6. Other experiments unchanged (still require 1 challenger)
# ----------------------------------------------------------------------
def test_e02_one_challenger_passes_invariant_shape():
    """E02 invariant check still expects exactly 1 challenger."""
    from course_work.model_improvement_v2.e02_runner import (
        build_e02_run_context,
        load_e02_config,
    )
    document = load_e02_config(PROJECT_ROOT)
    context = build_e02_run_context(PROJECT_ROOT, document)
    assert len(context.candidate_specs) == 1
    assert_context_invariants(context)


def test_e03_one_challenger_passes_invariant_shape():
    """E03 invariant check still expects exactly 1 challenger."""
    from course_work.model_improvement_v2.e03_runner import (
        build_e03_run_context,
        load_e03_config,
    )
    document = load_e03_config(PROJECT_ROOT)
    context = build_e03_run_context(PROJECT_ROOT, document)
    assert len(context.candidate_specs) == 1
    assert_context_invariants(context)


def test_e04_one_challenger_passes_invariant_shape():
    """E04 invariant check still expects exactly 1 challenger."""
    from course_work.model_improvement_v2.e04_runner import (
        build_e04_run_context,
        load_e04_config,
    )
    document = load_e04_config(PROJECT_ROOT)
    context = build_e04_run_context(PROJECT_ROOT, document)
    assert len(context.candidate_specs) == 1
    assert_context_invariants(context)


def test_e05_one_challenger_passes_invariant_shape():
    """E05 invariant check still expects exactly 1 challenger."""
    from course_work.model_improvement_v2.e05_runner import (
        build_e05_run_context,
        load_e05_config,
    )
    document = load_e05_config(PROJECT_ROOT)
    context = build_e05_run_context(PROJECT_ROOT, document)
    assert len(context.candidate_specs) == 1
    assert_context_invariants(context)


# ----------------------------------------------------------------------
# 7. Preflight-level OFF rejection (defense in depth)
# ----------------------------------------------------------------------
def test_e07_runner_rejects_off_challenger_at_preflight():
    """E07 runner preflight rejects scheduler_name=OFF in challengers."""
    from course_work.model_improvement_v2.e07_runner import (
        validate_e07_document,
        E07PreflightError,
    )
    document = deepcopy(_e07_document())
    # Mutate one challenger's scheduler_name to OFF
    document["challengers"][0]["scheduler_name"] = "OFF"
    document["challengers"][0]["scheduler_config"] = None
    document["challengers"][0]["config_fingerprint"] = "BROKEN"
    with pytest.raises(E07PreflightError, match="scheduler_name"):
        validate_e07_document(document, PROJECT_ROOT)
