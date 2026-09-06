# -*- coding: utf-8 -*-
"""Phase 59 — deterministic CSV / JSON / Markdown writers."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from . import constants as C


def write_csv(fp: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") if r.get(k, "") is not None else "") for k in fieldnames})


def write_json(fp: Path, obj) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def sha256_file(fp: Path) -> str:
    if not fp.is_file():
        return ""
    h = hashlib.sha256()
    h.update(fp.read_bytes())
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- Rounding helpers (mirror Phase 58) ----

def _is_missing(v) -> bool:
    if v is None or v == "":
        return True
    s = str(v).strip()
    if s == "" or s.upper() in ("N/A", "NA", "NAN", "NULL", "NONE"):
        return True
    return False


def fmt_display(v, unit: str) -> str:
    """Format a value with Phase 58 display rounding."""
    if _is_missing(v):
        return "N/A"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if unit in C.DISPLAY_PRECISION:
        dp = C.DISPLAY_PRECISION[unit]
        if dp == 0:
            return f"{int(round(f))}"
        return f"{f:.{dp}f}"
    return str(v)


def fmt_full(v, unit: str) -> str:
    """Full-precision string (used for CSV)."""
    if _is_missing(v):
        return "N/A"
    return str(v)


# ---- Markdown rendering ----

def write_markdown_table(fp: Path, title: str, columns: list[str],
                          rows: list[dict], notes: str = "") -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}\n"]
    if notes:
        parts.append(f"\n*{notes}*\n")
    parts.append("\n| " + " | ".join(columns) + " |")
    parts.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        cells = [str(r.get(c, "") or "").replace("|", "\\|") for c in columns]
        parts.append("| " + " | ".join(cells) + " |")
    fp.write_text("\n".join(parts) + "\n", encoding="utf-8")


# ---- Forbidden-phrase detection ----

WORD_RE = re.compile(r"\b[\w\-']+\b", flags=re.UNICODE)


def _phrase_count(text: str, phrase: str) -> int:
    """Count occurrences of phrase (case-insensitive, word-boundary where possible)."""
    if " " in phrase:
        return text.lower().count(phrase.lower())
    return len(WORD_RE.findall(text)) and sum(
        1 for w in WORD_RE.findall(text) if w.lower() == phrase.lower()
    )


def scan_forbidden_phrases(text: str) -> list[dict]:
    """Return a list of {phrase, count} entries with count >= 1."""
    out = []
    for ph in C.FORBIDDEN_PHRASES:
        n = _phrase_count(text, ph)
        if n > 0:
            out.append({"phrase": ph, "count": n})
    return out


def scan_review_phrases(text: str) -> list[dict]:
    out = []
    for ph in C.REVIEW_PHRASES:
        n = _phrase_count(text, ph)
        if n > 0:
            out.append({"phrase": ph, "count": n})
    return out


# ---- Numeric extraction ----

NUM_RE = re.compile(
    r"(?P<num>\d+\.\d+|\d+)\s*(?P<unit>Wh|R²|%|\bminutes\b|min|JSD|Cosine|Spearman|Cliff|Wasserstein|Wh|R\^2|units|\$)?",
    flags=re.IGNORECASE,
)


def extract_numbers(text: str) -> list[dict]:
    out = []
    for m in NUM_RE.finditer(text):
        num = m.group("num")
        unit = (m.group("unit") or "").strip()
        if "." not in num and len(num) > 5:
            # likely not a scientific number
            continue
        out.append({"num": num, "unit": unit, "span": m.span()})
    return out
