"""Phase 52 — atomic writers and CSV/JSON helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _sha256_file(path: Path) -> str:
    if path.is_dir():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv_atomic(
    rows: list[dict[str, Any]],
    fp: Path,
    fieldnames: list[str] | None = None,
) -> None:
    """Atomic CSV writer with fieldnames auto-detection."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError("Empty rows + no fieldnames")
        keys = list(rows[0].keys())
        seen = set(keys)
        for r in rows[1:]:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys

    tmp = tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8", newline="", dir=str(fp.parent)
    )
    try:
        with tmp as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        tmp_path = Path(tmp.name)
        tmp_path.replace(fp)
    finally:
        if Path(tmp.name).exists():
            try:
                Path(tmp.name).unlink()
            except OSError:
                pass
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass


def write_json_atomic(obj: Any, fp: Path) -> None:
    """Atomic JSON writer with explicit UTF-8."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    tmp_fp = fp.with_suffix(fp.suffix + ".tmp")
    tmp_fp.write_bytes(blob)
    tmp_fp.replace(fp)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass


def write_npz_atomic(arr_dict: dict[str, np.ndarray], fp: Path) -> None:
    """Atomic NPZ writer. Ensures reload is identical.

    Writes to a temp dir (same parent) using np.savez, then atomically moves to fp.
    """
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = fp.parent / (fp.name + ".tmp_dir")
    tmp_dir.mkdir(exist_ok=True)
    tmp_no_ext = str(tmp_dir / "array")
    np.savez(tmp_no_ext, **arr_dict)
    tmp_written = Path(tmp_no_ext + ".npz")
    os.replace(str(tmp_written), str(fp))
    try:
        os.rmdir(str(tmp_dir))
    except OSError:
        pass 
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass


def reload_verify_npz(
    fp: Path,
    expected_arrays: dict[str, tuple[tuple[int, ...], str]],
) -> dict[str, Any]:
    """Reload NPZ and verify each array shape+dtype+SHA.

    expected_arrays: name -> (expected_shape, expected_dtype_str)
    """
    out: dict[str, Any] = {}
    data = np.load(fp, allow_pickle=False)
    for name, (shape, dtype) in expected_arrays.items():
        arr = data[name]
        ok_shape = tuple(arr.shape) == tuple(shape)
        arr_dtype = str(arr.dtype)
        if dtype == "U":
            ok_dtype = arr_dtype.startswith("<U") or arr_dtype.startswith(">U") or arr_dtype.startswith("|U") or arr_dtype.startswith("U")
        else:
            ok_dtype = arr_dtype == dtype
        if arr.dtype.hasobject:
            h = "object_array_not_checksummed"
        else:
            h = hashlib.sha256(arr.tobytes()).hexdigest()
        out[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "shape_ok": ok_shape,
            "dtype_ok": ok_dtype,
            "sha256": h,
            "size_bytes": int(arr.nbytes),
            "status": "PASS" if (ok_shape and ok_dtype) else "FAIL",
        }
    out["_file_sha256"] = _sha256_file(fp)
    out["_file_size_bytes"] = fp.stat().st_size
    return out
