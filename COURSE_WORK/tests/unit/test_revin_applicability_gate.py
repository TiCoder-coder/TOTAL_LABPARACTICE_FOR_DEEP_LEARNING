"""Phase 40 — RN1 robustness tests for absent / duplicate Appliances feature order."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path('/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK')
sys.path.insert(0, str(ROOT / 'src'))

from course_work.models.revin import (
    resolve_revin_scope_from_feature_order,
)


def test_absent_appliances_returns_scope_none():
    """When Appliances is absent, scope resolution returns False with no scope object."""
    feature_order = ['lights', 'T1', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'weekend']
    ok, scope_or_none, reason = resolve_revin_scope_from_feature_order(feature_order)
    assert ok is False
    assert scope_or_none is None
    assert 'Appliances' in reason


def test_duplicate_appliances_raises():
    """Duplicate Appliances should fail at resolve time with ValueError."""
    feature_order = ['Appliances', 'lights', 'Appliances']
    with pytest.raises(ValueError):
        resolve_revin_scope_from_feature_order(feature_order)


def test_exactly_one_appliances_passes():
    feature_order = ['lights', 'Appliances', 'T1', 'hour_sin']
    ok, scope_or_none, reason = resolve_revin_scope_from_feature_order(feature_order)
    assert ok is True
    assert scope_or_none is not None
    assert scope_or_none.target_channel_name == 'Appliances'
    assert scope_or_none.revin_channel_count == 3  # lights, Appliances, T1