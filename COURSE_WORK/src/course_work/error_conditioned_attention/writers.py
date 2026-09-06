"""Phase 56 - CSV/JSON writers with NaN-safe serialization."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _safe(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return v
    return v


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            row_out = {c: _safe(r.get(c, "")) for c in columns}
            writer.writerow(row_out)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=_default, ensure_ascii=False), encoding="utf-8")
