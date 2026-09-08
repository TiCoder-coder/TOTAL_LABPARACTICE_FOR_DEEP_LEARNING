"""Phase 54 — atomic writers.

Used by all Phase 54 sub-phases. CSV / JSON writers with deterministic encoding.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


def write_csv_atomic(fp: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    """Write CSV atomically (write tmp, rename). Creates parent dirs."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = fp.with_suffix(fp.suffix + ".tmp")
    with tmp_fp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    os.replace(tmp_fp, fp)


def write_json_atomic(fp: Path, payload: dict, *, indent: int = 2) -> None:
    """Write JSON atomically."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = fp.with_suffix(fp.suffix + ".tmp")
    with tmp_fp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False, sort_keys=False)
    os.replace(tmp_fp, fp)


def read_csv(fp: Path) -> list[dict[str, str]]:
    with fp.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(fp: Path) -> dict:
    return json.loads(fp.read_text(encoding="utf-8"))


def append_csv(fp: Path, row: dict, fieldnames: list[str]) -> None:
    """Append a single row to a CSV. Creates file with header if missing."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    new = not fp.exists()
    with fp.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerow(row)
