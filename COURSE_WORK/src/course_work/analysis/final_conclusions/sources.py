# -*- coding: utf-8 -*-
"""Phase 59 — read-only source loaders over Phase 58 frozen artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from . import constants as C


def _read_csv(fp: Path) -> list[dict[str, str]]:
    if not fp.is_file():
        return []
    with fp.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(fp: Path) -> dict:
    if not fp.is_file():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def _sha256_file(fp: Path) -> str:
    if not fp.is_file():
        return ""
    h = hashlib.sha256()
    h.update(fp.read_bytes())
    return h.hexdigest()


class FrozenSources59:
    """Load frozen Phase 58 sources for Phase 59 (read-only)."""

    def __init__(self, project_root: Path | str):
        self.root = Path(project_root).resolve()
        self.art58 = self.root / "artifacts" / "final_tables"
        self.tables_dir = self.art58 / "tables/csv"

    # ---- Phase 58 metadata ----
    @property
    def signoff(self) -> dict:
        return _read_json(self.art58 / "phase_58_signoff.json")

    @property
    def handoff(self) -> dict:
        return _read_json(self.art58 / "phase59_final_conclusions_handoff.json")

    @property
    def summary(self) -> dict:
        return _read_json(self.art58 / "final_tables_summary.json")

    @property
    def inventory(self) -> dict:
        return _read_json(self.art58 / "final_table_inventory.json")

    @property
    def contract(self) -> dict:
        return _read_json(self.art58 / "final_tables_contract.json")

    # ---- Phase 58 machine-readable evidence ----
    @property
    def source_ledger(self) -> list[dict[str, str]]:
        return _read_csv(self.art58 / "final_table_source_ledger.csv")

    @property
    def cell_lineage(self) -> list[dict[str, str]]:
        return _read_csv(self.art58 / "final_table_cell_lineage.csv")

    @property
    def claim_traceability(self) -> list[dict[str, str]]:
        return _read_csv(self.art58 / "table_claim_traceability.csv")

    @property
    def findings58(self) -> list[dict[str, str]]:
        return _read_csv(self.art58 / "final_tables_findings.csv")

    # ---- Main table content ----
    def ft(self, tid: str) -> list[dict[str, str]]:
        return _read_csv(self.tables_dir / f"{tid}_rows.csv")

    def fa(self, tid: str) -> list[dict[str, str]]:
        return _read_csv(self.tables_dir / f"{tid}_rows.csv")

    @property
    def main_tables(self) -> dict[str, list[dict[str, str]]]:
        return {tid: self.ft(tid) for tid in (
            "FT01", "FT02", "FT03", "FT04", "FT05",
            "FT06", "FT07", "FT08", "FT09", "FT10",
        )}

    @property
    def appendix_tables(self) -> dict[str, list[dict[str, str]]]:
        return {tid: self.fa(tid) for tid in (
            "FA01", "FA02", "FA03", "FA04", "FA05", "FA06",
            "FA07", "FA08", "FA09", "FA10", "FA11", "FA12",
        )}

    # ---- Upstream frozen provenance ----
    @property
    def final_lock_sha(self) -> str:
        return self.signoff.get("final_lock_sha256", "") or ""

    @property
    def test_pop_sha(self) -> str:
        return self.signoff.get("final_test_population_sha256", "") or ""

    @property
    def catalog_md(self) -> str:
        fp = self.art58 / "FINAL_TABLE_CATALOG.md"
        return fp.read_text(encoding="utf-8") if fp.is_file() else ""

    @property
    def official_seeds(self) -> list[int]:
        return list(C.OFFICIAL_SEEDS)

    def all_frozen_artifacts(self) -> dict[str, str]:
        """Return SHA256 fingerprints of every frozen Phase 58 artifact."""
        out: dict[str, str] = {}
        targets = [
            "phase_58_signoff.json",
            "final_tables_summary.json",
            "final_table_inventory.json",
            "final_table_source_ledger.csv",
            "final_table_cell_lineage.csv",
            "table_claim_traceability.csv",
            "final_tables_findings.csv",
            "FINAL_TABLE_CATALOG.md",
        ]
        for name in targets:
            fp = self.art58 / name
            out[name] = _sha256_file(fp)
        # FA12 caveats
        for fp in self.tables_dir.glob("FA12_*.csv"):
            out[fp.name] = _sha256_file(fp)
        # FT/FA tables
        for fp in self.tables_dir.glob("*.csv"):
            out[fp.name] = _sha256_file(fp)
        return out
