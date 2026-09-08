"""Unit tests for Phase 44 failure-resume simulation (TASK 11)."""
from __future__ import annotations

from course_work.rolling_origin.failure_resume import (
    FailureScenario,
    simulate_failure,
    run_all_failure_simulations,
)


def test_during_stage_a_new_run_id_required() -> None:
    a = simulate_failure(FailureScenario.DURING_STAGE_A)
    assert a.status == "PASS"
    assert a.new_run_id_required is True
    assert not a.orphan_records_left


def test_after_stage_a_before_b_no_new_run_id() -> None:
    a = simulate_failure(FailureScenario.AFTER_STAGE_A_BEFORE_B)
    assert a.status == "PASS"
    assert a.new_run_id_required is False
    assert not a.orphan_records_left


def test_during_stage_b_new_run_id_required() -> None:
    a = simulate_failure(FailureScenario.DURING_STAGE_B)
    assert a.status == "PASS"
    assert a.new_run_id_required is True
    assert not a.orphan_records_left


def test_after_stage_b_before_c_resume_without_new_id() -> None:
    a = simulate_failure(FailureScenario.AFTER_STAGE_B_BEFORE_C)
    assert a.status == "PASS"
    assert a.new_run_id_required is False


def test_during_finalization_pass() -> None:
    a = simulate_failure(FailureScenario.DURING_FINALIZATION)
    assert a.status == "PASS"
    assert a.new_run_id_required is False


def test_all_five_scenarios_pass() -> None:
    audits = run_all_failure_simulations()
    assert len(audits) == 5
    for a in audits:
        assert a.status == "PASS"
        assert a.new_run_id_required in {True, False}
