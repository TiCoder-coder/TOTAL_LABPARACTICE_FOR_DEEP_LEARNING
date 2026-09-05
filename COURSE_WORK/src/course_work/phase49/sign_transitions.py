from __future__ import annotations

from datetime import datetime
from typing import Any

from .autocorrelation import PHASE49_CADENCE_MINUTES


SIGN_LABELS = ("UNDERPREDICTION", "OVERPREDICTION", "EXACT")


def sign_transitions_for_seed(
    signs: list[str],
    timestamps: list[datetime],
    cadence_minutes: int = PHASE49_CADENCE_MINUTES,
) -> dict[str, Any]:
    """Count transitions between consecutive observations EXACTLY 10 minutes apart.

    Pairs with delta != cadence_minutes are excluded (no bridging across gaps).
    No smoothing, no epsilon-zero, exact cadence required.
    """
    if len(signs) != len(timestamps):
        raise ValueError("signs and timestamps must have equal length")
    for s in signs:
        if s not in SIGN_LABELS:
            raise ValueError(f"Invalid sign label: {s!r}")

    counts: dict[tuple[str, str], int] = {}
    for a in SIGN_LABELS:
        for b in SIGN_LABELS:
            counts[(a, b)] = 0

    eligible_pairs = 0
    for i in range(1, len(signs)):
        delta_minutes = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60.0
        if delta_minutes != cadence_minutes:
            continue
        eligible_pairs += 1
        counts[(signs[i - 1], signs[i])] += 1

    rows: list[dict[str, Any]] = []
    for a in SIGN_LABELS:
        out_total = sum(counts[(a, b)] for b in SIGN_LABELS)
        for b in SIGN_LABELS:
            c = counts[(a, b)]
            prob = float(c) / float(out_total) if out_total > 0 else float("nan")
            rows.append(
                {
                    "from_sign": a,
                    "to_sign": b,
                    "count": c,
                    "outgoing_total_from_a": out_total,
                    "probability_given_from_a": prob,
                }
            )

    return {
        "eligible_pairs": eligible_pairs,
        "transition_rows": rows,
        "transition_count_table": {
            f"{a}->{b}": counts[(a, b)] for a in SIGN_LABELS for b in SIGN_LABELS
        },
    }


def sign_transition_summary_row(
    seed: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    probabilities = [r["probability_given_from_a"] for r in result["transition_rows"]]
    return {
        "seed": seed,
        "eligible_pairs": result["eligible_pairs"],
        "transition_count_total": sum(result["transition_count_table"].values()),
        "probabilities_sum_per_from_sign": [
            sum(r["probability_given_from_a"] for r in result["transition_rows"] if r["from_sign"] == a)
            for a in SIGN_LABELS
        ],
        "cadence_minutes": PHASE49_CADENCE_MINUTES,
        "epsilon_used": False,
        "smoothing_used": False,
    }
