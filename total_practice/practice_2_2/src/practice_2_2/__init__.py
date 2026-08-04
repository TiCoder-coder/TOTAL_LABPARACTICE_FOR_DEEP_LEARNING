"""Canonical Practice 2.2 package with legacy import compatibility.

Phase 6 makes ``practice_2_2.*`` the recommended import namespace and the new
layout the resource authority. Adding the old project root to ``sys.path``
preserves shared implementation imports without changing training behavior.
"""

import sys

from .paths import get_practice_2_root

_LEGACY_PRACTICE_2_ROOT = get_practice_2_root()
legacy_root = str(_LEGACY_PRACTICE_2_ROOT)
if legacy_root not in sys.path:
    sys.path.insert(0, legacy_root)


def compatibility_status():
    """Return read-only ownership information for migration verification."""

    return {
        "phase": 6,
        "old_import_root": "processing_own_phase",
        "new_import_root": "practice_2_2",
        "legacy_shared_source": str(
            (_LEGACY_PRACTICE_2_ROOT / "processing_own_phase").resolve()
        ),
        "canonical_entrypoint_switched": True,
    }
