# -*- coding: utf-8 -*-
"""Phase 58 - writers (CSV / Markdown / LaTeX / JSON) with safe NaN handling.

All writers preserve upstream full precision. Display rounding is applied at
output time only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .constants import DISPLAY_PRECISION, MISSING_LABEL, NEGATIVE_ZERO_FIX


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if s == "" or s.upper() in ("N/A", "NA", "NAN", "NONE", "NULL"):
        return True
    try:
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            return True
    except Exception:
        return False
    return False


def _fmt_full(v: Any) -> str:
    """Format full-precision numeric value as a stable string."""
    if _is_missing(v):
        return ""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return ""
        # Use repr() for full precision
        return repr(f)
    except Exception:
        return str(v).strip()


def _fmt_display(v: Any, unit_family: str) -> str:
    """Format display value with locked precision; emit MISSING_LABEL if missing."""
    if _is_missing(v):
        return MISSING_LABEL
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return MISSING_LABEL
        prec = DISPLAY_PRECISION.get(unit_family, 3)
        out = f"{f:.{prec}f}"
        if NEGATIVE_ZERO_FIX:
            # Replace -0.000... with 0.000...
            if float(out) == 0.0 and f < 0:
                out = "-" .replace("-", "")
                if "." in out:
                    out = out.lstrip("-")
        return out
    except Exception:
        return str(v).strip()


def write_csv(fp: Path, rows: list[dict], cols: list[str]) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {c: _fmt_full(r.get(c, "")) for c in cols}
            w.writerow(out)


def write_json(fp: Path, obj: Any) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        try:
            return float(o)
        except Exception:
            return str(o)

    fp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default),
                  encoding="utf-8")


def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- Markdown ----------

def _md_escape(text: str) -> str:
    """Escape vertical bar and backslash for Markdown table cells."""
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def write_markdown(fp: Path, rows: list[dict], cols: list[str],
                   column_units: dict[str, str] | None = None,
                   title: str = "",
                   caption: str = "",
                   footer_notes: list[str] | None = None) -> None:
    """Write a Markdown table.

    column_units: maps each column key -> unit family (Wh / r2 / minutes / etc.)
                  used to apply display rounding.
    """
    fp.parent.mkdir(parents=True, exist_ok=True)
    column_units = column_units or {}
    out: list[str] = []
    if title:
        out.append(f"## {title}\n")
    if caption:
        out.append(f"_{caption}_\n")
    out.append("| " + " | ".join(_md_escape(c) for c in cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells = []
        for c in cols:
            unit = column_units.get(c, "")
            cells.append(_md_escape(_fmt_display(r.get(c, ""), unit)))
        out.append("| " + " | ".join(cells) + " |")
    if footer_notes:
        out.append("")
        for note in footer_notes:
            out.append(f"> {note}")
    fp.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------- LaTeX ----------

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


def _latex_escape(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(_LATEX_SPECIAL.get(ch, ch))
    return "".join(out)


def write_latex(fp: Path, rows: list[dict], cols: list[str],
                column_units: dict[str, str] | None = None,
                title: str = "",
                caption: str = "",
                footer_notes: list[str] | None = None) -> None:
    column_units = column_units or {}
    fp.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    out.append(r"\begin{table}[h!]")
    out.append(r"\centering")
    if title:
        out.append(r"\caption{" + _latex_escape(title) + r"}")
    if caption:
        out.append(r"\label{tab:" + _latex_escape(title.lower().replace(" ", "_")) + r"}")
    cols_spec = "l" + ("r" * (len(cols) - 1))
    out.append(r"\begin{tabular}{" + cols_spec + r"}")
    out.append(r"\toprule")
    out.append(" & ".join(_latex_escape(c) for c in cols) + r" \\")
    out.append(r"\midrule")
    for r in rows:
        cells = []
        for c in cols:
            unit = column_units.get(c, "")
            cells.append(_latex_escape(_fmt_display(r.get(c, ""), unit)))
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    if footer_notes:
        out.append(r"\vspace{4pt}")
        for note in footer_notes:
            out.append(r"\footnotesize{" + _latex_escape(note) + r"}\\")
    out.append(r"\end{table}")
    fp.write_text("\n".join(out) + "\n", encoding="utf-8")
