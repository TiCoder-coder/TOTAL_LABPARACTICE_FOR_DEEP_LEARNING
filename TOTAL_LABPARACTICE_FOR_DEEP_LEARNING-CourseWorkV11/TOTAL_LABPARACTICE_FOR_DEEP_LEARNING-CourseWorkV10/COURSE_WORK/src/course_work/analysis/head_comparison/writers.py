"""Phase 55 - CSV / JSON writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.8f}"
    return str(v)


def write_dictlist_csv(rows: list[dict[str, Any]], fp: Path, fields: list[str] | None = None) -> None:
    if not rows:
        fp.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields = list(rows[0].keys())
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})


def write_json(d: dict[str, Any], fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)
