"""Phase 57 - writers (CSV / JSON utilities with safe NaN/Inf handling)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import math


def _safe(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return v
    return v


def write_csv(fp: Path, rows: list[dict[str, Any]], columns: list[str]) -> int:
    fp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with fp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for row in rows:
            w.writerow([_safe(row.get(c, "")) for c in columns])
            n += 1
    return n


def write_json(fp: Path, obj: dict) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_csv(fp: Path, rows: list[dict[str, Any]], columns: list[str], header: bool = True) -> int:
    fp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    is_new = not fp.exists()
    with fp.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if is_new and header:
            w.writerow(columns)
        for row in rows:
            w.writerow([_safe(row.get(c, "")) for c in columns])
            n += 1
    return n
