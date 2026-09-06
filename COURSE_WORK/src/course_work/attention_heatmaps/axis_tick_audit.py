"""Phase 53 — axis tick audit (lag-tick policy vs frozen Phase52 position map).

`attention_heatmap_axis_tick_audit.csv` per canonical schema §132:

  lookback
  requested_lag_steps
  included
  position_idx0
  lag_minutes
  x_tick_label
  y_tick_label
  mapping_source
  status

We trust the frozen Phase52 relative position map (lag_steps_p = L - p,
H=1). 1 ≤ lag ≤ L is the include rule. Always include oldest lag if needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .sources import (
    CADENCE_MINUTES,
    INCLUDE_OLDEST_TICK,
    LAG_TICK_STEPS_REQUESTED,
    LOOKBACK,
)


def compute_axis_tick_audit(project_root: Path) -> list[dict]:
    """Read frozen Phase52 relative-position map and compute lag tick audit."""
    fp = project_root / "artifacts/attention_extraction/attention_relative_position_map.csv"
    rpm = {}  # position_idx0 -> lag_minutes, lag_steps
    with fp.open() as f:
        header = f.readline().strip().split(",")
        # find indices
        idx_pos = header.index("position_idx0")
        idx_lag_steps = header.index("lag_steps_from_forecast_target")
        idx_lag_min = header.index("lag_minutes_from_forecast_target")
        idx_status = header.index("status")
        for line in f:
            fields = line.rstrip().split(",")
            if len(fields) <= max(idx_pos, idx_lag_steps, idx_lag_min):
                continue
            pos = int(fields[idx_pos])
            lag_steps = int(fields[idx_lag_steps])
            lag_min = int(fields[idx_lag_min])
            rpm[pos] = {
                "lag_steps": lag_steps,
                "lag_minutes": lag_min,
                "status": fields[idx_status],
            }

    rows = []
    for requested in LAG_TICK_STEPS_REQUESTED:
        included = 1 <= requested <= LOOKBACK
        # Find position that has lag_steps == requested
        position_idx = None
        for pos, d in rpm.items():
            if d["lag_steps"] == requested:
                position_idx = pos
                break
        if position_idx is None:
            rows.append({
                "lookback": LOOKBACK,
                "requested_lag_steps": requested,
                "included": False,
                "position_idx0": None,
                "lag_minutes": None,
                "x_tick_label": f"lag {requested} (excluded; > LOOKBACK)",
                "y_tick_label": f"lag {requested} (excluded; > LOOKBACK)",
                "mapping_source": "PHASE52_RELATIVE_POSITION_MAP",
                "status": "EXCLUDED",
            })
            continue
        d = rpm[position_idx]
        rows.append({
            "lookback": LOOKBACK,
            "requested_lag_steps": requested,
            "included": True,
            "position_idx0": position_idx,
            "lag_minutes": d["lag_minutes"],
            "x_tick_label": f"lag {requested} ({d['lag_minutes']} min)",
            "y_tick_label": f"lag {requested} ({d['lag_minutes']} min)",
            "mapping_source": "PHASE52_RELATIVE_POSITION_MAP",
            "status": "OK" if d["status"] == "OK" else "FAIL",
        })
    # Always include oldest
    if INCLUDE_OLDEST_TICK:
        oldest_pos = 0
        d = rpm.get(oldest_pos, {"lag_steps": LOOKBACK, "lag_minutes": LOOKBACK * CADENCE_MINUTES, "status": "OK"})
        rows.append({
            "lookback": LOOKBACK,
            "requested_lag_steps": d["lag_steps"],
            "included": True,
            "position_idx0": oldest_pos,
            "lag_minutes": d["lag_minutes"],
            "x_tick_label": f"lag {d['lag_steps']} ({d['lag_minutes']} min; oldest)",
            "y_tick_label": f"lag {d['lag_steps']} ({d['lag_minutes']} min; oldest)",
            "mapping_source": "PHASE52_RELATIVE_POSITION_MAP",
            "status": "OK",
        })
    # Always include newest (position L-1)
    newest_pos = LOOKBACK - 1
    d = rpm.get(newest_pos, {"lag_steps": 1, "lag_minutes": 10, "status": "OK"})
    rows.append({
        "lookback": LOOKBACK,
        "requested_lag_steps": d["lag_steps"],
        "included": True,
        "position_idx0": newest_pos,
        "lag_minutes": d["lag_minutes"],
        "x_tick_label": f"lag {d['lag_steps']} ({d['lag_minutes']} min; newest)",
        "y_tick_label": f"lag {d['lag_steps']} ({d['lag_minutes']} min; newest)",
        "mapping_source": "PHASE52_RELATIVE_POSITION_MAP",
        "status": "OK",
    })
    return rows


def write_axis_tick_audit(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit("Phase 53 axis_tick_audit is a library — import it from orchestrator.")
