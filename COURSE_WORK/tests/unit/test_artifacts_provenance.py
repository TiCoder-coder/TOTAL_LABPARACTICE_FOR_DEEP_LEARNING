"""Regression tests for provenance metadata handling in artifact writing.

These tests verify:
  - Same scientific JSON content + different created_at does NOT raise FileExistsError
  - Different scientific JSON content still raises FileExistsError
  - Binary artifacts (.joblib) still use strict byte comparison
  - The _strip_provenance_metadata function works correctly
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from course_work.utils.artifacts import (
    _strip_provenance_metadata,
    atomic_write_bytes,
    canonical_json_bytes,
    write_bytes_once_or_verify,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


class TestStripProvenanceMetadata:
    def test_strips_top_level_created_at(self):
        value = {
            "scientific_field": "important",
            "created_at": "2026-09-01T10:00:00+00:00",
            "nested": {
                "also_important": 123,
                "created_at": "2026-09-01T10:00:00+00:00",
            },
        }
        result = _strip_provenance_metadata(value)
        assert "created_at" not in result
        assert result["scientific_field"] == "important"
        assert "created_at" not in result["nested"]
        assert result["nested"]["also_important"] == 123

    def test_preserves_all_non_provenance_fields(self):
        value = {
            "scaling_version": "FINAL_SCALING-v1",
            "fit_row_count": 16630,
            "x_sha256": "abc123",
            "y_sha256": "def456",
            "population_fingerprint": "fp123",
            "created_at": "2026-09-01T10:00:00+00:00",
            "list_field": [1, 2, 3],
        }
        result = _strip_provenance_metadata(value)
        assert "created_at" not in result
        assert result["scaling_version"] == "FINAL_SCALING-v1"
        assert result["fit_row_count"] == 16630
        assert result["list_field"] == [1, 2, 3]

    def test_handles_list_values(self):
        value = [{"created_at": "2026-01-01", "value": 1}, {"created_at": "2026-01-02", "value": 2}]
        result = _strip_provenance_metadata(value)
        assert all("created_at" not in item for item in result)
        assert result[0]["value"] == 1
        assert result[1]["value"] == 2

    def test_handles_primitive_values(self):
        assert _strip_provenance_metadata("string") == "string"
        assert _strip_provenance_metadata(123) == 123
        assert _strip_provenance_metadata(None) is None
        assert _strip_provenance_metadata(True) is True


class TestWriteJsonOnceOrVerifyProvenance:
    def test_same_content_different_created_at_does_not_raise(self):
        """Re-materializing with a new wall-clock created_at must NOT raise
        FileExistsError when the scientific payload is unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"

            # First write.
            value1 = {
                "scientific_field": "unchanged",
                "fit_row_count": 16630,
                "created_at": "2026-09-01T10:00:00+00:00",
            }
            write_json_once_or_verify(path, value1)

            # Second write with same scientific content but new timestamp.
            value2 = {
                "scientific_field": "unchanged",
                "fit_row_count": 16630,
                "created_at": "2026-09-01T11:00:00+00:00",  # different!
            }
            # Re-materialization with identical scientific content must NOT raise.
            # The function keeps the existing artifact on disk (no unnecessary rewrite).
            result = write_json_once_or_verify(path, value2)
            assert result == path
            assert path.exists()

            # Content on disk is preserved (no unnecessary overwrite).
            stored = json.loads(path.read_bytes())
            assert stored["created_at"] == "2026-09-01T10:00:00+00:00"  # original kept
            assert stored["scientific_field"] == "unchanged"

    def test_different_scientific_content_still_raises(self):
        """A change in the scientific payload MUST still raise FileExistsError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"

            value1 = {
                "scaling_version": "FINAL_SCALING-v1",
                "fit_row_count": 16630,
                "created_at": "2026-09-01T10:00:00+00:00",
            }
            write_json_once_or_verify(path, value1)

            value2 = {
                "scaling_version": "FINAL_SCALING-v1",
                "fit_row_count": 16631,  # changed!
                "created_at": "2026-09-01T11:00:00+00:00",
            }
            with pytest.raises(FileExistsError, match="Signed artifact differs"):
                write_json_once_or_verify(path, value2)

    def test_binary_artifacts_use_strict_byte_comparison(self):
        """Binary artifacts (.joblib, .pt) must NOT be affected by this fix —
        they use strict byte comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # First write.
            content1 = b"model_v1"
            write_bytes_once_or_verify(path, content1)

            # Second write with different bytes must raise.
            content2 = b"model_v2"
            with pytest.raises(FileExistsError, match="Signed artifact differs"):
                write_bytes_once_or_verify(path, content2)

            # Same bytes must NOT raise.
            result = write_bytes_once_or_verify(path, content1)
            assert result == path

    def test_nested_created_at_in_lists_ignored(self):
        """created_at inside list items is also stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"

            value1 = {
                "bundles": [
                    {"bundle_id": "XSCALER", "created_at": "2026-09-01T10:00:00+00:00"},
                    {"bundle_id": "YSCALER", "created_at": "2026-09-01T10:00:00+00:00"},
                ],
                "created_at": "2026-09-01T10:00:00+00:00",
            }
            write_json_once_or_verify(path, value1)

            value2 = {
                "bundles": [
                    {"bundle_id": "XSCALER", "created_at": "2026-10-01T10:00:00+00:00"},
                    {"bundle_id": "YSCALER", "created_at": "2026-10-01T10:00:00+00:00"},
                ],
                "created_at": "2026-10-01T10:00:00+00:00",
            }
            # Re-materialization with only created_at changes must NOT raise.
            # Existing artifact is kept (no unnecessary overwrite).
            result = write_json_once_or_verify(path, value2)
            assert result == path

    def test_text_artifacts_still_use_byte_comparison(self):
        """Text artifacts also use strict comparison (created_at only stripped from JSON)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.csv"

            value1 = "header\nrow1\n"
            write_text_once_or_verify(path, value1)

            value2 = "header\nrow2\n"
            with pytest.raises(FileExistsError, match="Signed artifact differs"):
                write_text_once_or_verify(path, value2)

            # Same text does not raise.
            result = write_text_once_or_verify(path, value1)
            assert result == path
