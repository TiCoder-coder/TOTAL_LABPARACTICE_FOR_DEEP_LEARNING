"""Phase 51-C — frozen-contract sort-key compliance audit.

This module verifies that the active ranking code uses ONLY the
sort keys registered in the frozen selection contract, ONLY inside
the actual ``_key`` lambdas / ``sorted(... key=...)`` constructions.
References in field-name lists, dictionaries, or row payload dicts
are NOT sort keys.

FROZEN (worst_error_selection_contract.json):
  primary_ranking_metric:  absolute_error_wh   (per-seed) /
                            mean_abs_error_wh   (shared)
  primary_ranking_direction: DESC
  tie_break: target_id ASC
"""
from __future__ import annotations

import re
from pathlib import Path


# Sort-key fields permitted by the frozen contract:
PERMITTED_SORT_KEYS = {
    "absolute_error_wh",
    "mean_abs_error_wh",
    "target_id",
}

# Fields that must never appear as a tier in ranking sort keys.
FORBIDDEN_SORT_KEYS = {
    "target_timestamp",
    "timestamp",
    "ts",
    "y_true_wh",
    "y_pred_wh",
    "residual_wh",
    "squared_error_wh2",
}

RANKING_MODULES = (
    "src/course_work/phase51/ranking.py",
    "src/course_work/phase51/signed_ranking.py",
)


def _module_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _extract_key_construction_blocks(content: str) -> list[str]:
    """Pull out the bodies of every ``_key`` function or lambda.

    These are the only sites where a sort key (a tuple/values used in
    ``sorted(..., key=...)``) is being declared. The function maps:
      def _key(r):
        ...
      return ...

    and the inline ``sorted(rows, key=lambda r: (...))`` cases.
    """
    blocks: list[str] = []

    # Case 1: def _key(r): ... return ...  — match balanced by tracking
    # the ``return`` statement at the same indentation level (top of def body).
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)def\s+_key\s*\(\s*r\s*\)\s*:\s*$", line)
        if m:
            indent = m.group(1)
            block_lines = [line]
            j = i + 1
            # Capture lines until ``return`` at the SAME or smaller indent than ``def``
            return_idx = None
            for k in range(j, len(lines)):
                ln = lines[k]
                if re.match(r"^\s*return\s", ln) and (
                    not ln.startswith(indent + "    ")
                ):
                    return_idx = k
                    break
            if return_idx is None:
                return_idx = len(lines)
            block_lines.extend(lines[j:return_idx + 1])
            blocks.append("\n".join(block_lines))
            i = return_idx + 1
            continue
        i += 1

    # Case 2: sorted(rows, key=lambda r: (...))  — match the lambda body
    # up to the balanced closing parenthesis.
    lambda_pattern = re.compile(r"sorted\([^)]*?key\s*=\s*lambda[^:]*:\s*")
    for m in lambda_pattern.finditer(content):
        start = m.end()
        depth = 1
        k = start
        while k < len(content) and depth > 0:
            ch = content[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        blocks.append(content[start:k])

    return blocks


def audit_no_forbidden_sort_keys() -> dict[str, dict[str, list[str]]]:
    """Scan ranking modules for forbidden sort-key field REFERENCES inside
    every ``_key`` function/lambda body.

    Returns a mapping {module: {"forbidden": [...], "permitted": [...]}}.
    """
    root = _module_root()
    findings: dict[str, dict[str, list[str]]] = {}
    for rel in RANKING_MODULES:
        fp = root / rel
        content = fp.read_text(encoding="utf-8")
        blocks = _extract_key_construction_blocks(content)
        combined = "\n".join(blocks)

        forbidden_hits: list[str] = []
        for fld in FORBIDDEN_SORT_KEYS:
            pattern = r"\b" + re.escape(fld) + r"\b"
            if re.search(pattern, combined):
                forbidden_hits.append(fld)

        permitted_hits: list[str] = []
        for fld in PERMITTED_SORT_KEYS:
            pattern = r"\b" + re.escape(fld) + r"\b"
            if re.search(pattern, combined):
                permitted_hits.append(fld)

        findings[rel] = {
            "forbidden": forbidden_hits,
            "permitted": permitted_hits,
        }
    return findings


def verify_frozen_sort_keys_used() -> bool:
    """Return True iff no forbidden sort-key field appears in any ranker's
    ``_key`` body and at least one permitted key is referenced in each ranker.
    """
    findings = audit_no_forbidden_sort_keys()
    for module_result in findings.values():
        if module_result["forbidden"]:
            return False
        if not module_result["permitted"]:
            return False
    return True
