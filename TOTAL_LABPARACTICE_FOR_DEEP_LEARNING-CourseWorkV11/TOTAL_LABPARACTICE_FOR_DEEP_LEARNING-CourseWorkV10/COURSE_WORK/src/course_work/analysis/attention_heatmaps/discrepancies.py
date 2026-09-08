"""Phase 53 — discrepancy taxonomy + status model.

Per canonical Phase 53 plan §158 / §159:

* RESOLVED: documented and fixed during Phase 53 execution
* DOCUMENTED: known but not a scientific failure
* OPEN: any open scientific-invalidating item ⇒ FAIL

The discrepancy report is `attention_heatmap_discrepancies.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_discrepancies(items: list[dict], output_json: Path) -> dict:
    """Write the discrepancies JSON and return it for use in summary.

    Each item must have: id, category, status, description, resolution_notes.
    """
    payload = {
        "items": items,
        "n_resolved": sum(1 for it in items if it["status"] == "RESOLVED"),
        "n_documented": sum(1 for it in items if it["status"] == "DOCUMENTED"),
        "n_open": sum(1 for it in items if it["status"] == "OPEN"),
        "critical_open_ids": [
            it["id"] for it in items
            if it["status"] == "OPEN" and it.get("critical", False)
        ],
        "high_scientific_open_ids": [
            it["id"] for it in items
            if it["status"] == "OPEN" and it.get("high_scientific", False)
        ],
        "status": "PASS",  
    }
    if any(it.get("critical", False) and it["status"] == "OPEN" for it in items):
        payload["status"] = "FAIL"
    elif any(it.get("high_scientific", False) and it["status"] == "OPEN" for it in items):
        payload["status"] = "FAIL"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    raise SystemExit("Phase 53 discrepancies is a library — import it from orchestrator.")
