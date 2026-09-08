import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_json_bytes(value: Any) -> bytes:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return f"{content}\n".encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def write_bytes_once_or_verify(path: Path, content: bytes) -> Path:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Signed artifact differs from requested content: {path}")
        return path
    return atomic_write_bytes(path, content)


def write_bytes_once_or_verify_permissive(
    path: Path,
    content: bytes,
    statistics_compare_fn=None,
) -> Path:
    """Write bytes atomically, or verify them if the file already exists.

    When bytes differ (e.g., sklearn version drift in joblib serialization),
    this function falls back to a statistics comparison. If the caller
    provides a ``statistics_compare_fn(bytes) -> bool`` that returns True,
    the existing file is accepted as equivalent and this function returns
    without overwriting.

    This allows re-materialization in environments with different library
    versions while still protecting against unintended overwrites of
    scientifically different artifacts.

    Raises:
        FileExistsError: if bytes differ AND statistics comparison is absent or returns False.
    """
    if path.exists():
        if path.read_bytes() == content:
            return path
        if statistics_compare_fn is not None and statistics_compare_fn(content):
            return path
        raise FileExistsError(
            f"Signed artifact bytes differ and statistics comparison failed: {path}"
        )
    return atomic_write_bytes(path, content)


def _strip_provenance_metadata(value: Any) -> Any:
    """Remove provenance-only fields from an artifact value for comparison.

    Provenance fields carry no scientific identity and are regenerated on every
    materialization.  They must not cause a false mismatch when the actual
    scientific payload is unchanged.

    Provenance fields (excluded from signed-artifact equality check):
      - created_at: wall-clock timestamp of materialization
    """
    if isinstance(value, dict):
        return {k: _strip_provenance_metadata(v) for k, v in value.items() if k != "created_at"}
    if isinstance(value, list):
        return [_strip_provenance_metadata(item) for item in value]
    return value


def write_json_once_or_verify(path: Path, value: Any) -> Path:
    canonical = canonical_json_bytes(value)
    if path.exists():
        existing = json.loads(path.read_bytes())
        incoming = json.loads(canonical)
        existing_stripped = canonical_json_bytes(_strip_provenance_metadata(existing))
        incoming_stripped = canonical_json_bytes(_strip_provenance_metadata(incoming))
        if existing_stripped != incoming_stripped:
            raise FileExistsError(f"Signed artifact differs from requested content: {path}")
        return path
    return atomic_write_bytes(path, canonical)


def write_json_once_or_verify_permissive(
    path: Path,
    value: Any,
    compare_keys: list[str] | None = None,
) -> Path:
    """Write JSON atomically, or verify it if the file already exists.

    When the file exists but the content differs, this function strips
    provenance-only fields (created_at) and, if a ``compare_keys`` whitelist
    is provided, only compares those specific fields for scientific equivalence.
    If compare_keys is None, falls back to the standard provenance-stripped
    comparison.

    This allows re-writing metadata-only updates (like lookback_steps=72
    on an existing manifest whose scientific content is unchanged).
    """
    canonical = canonical_json_bytes(value)
    if path.exists():
        existing = json.loads(path.read_bytes())
        incoming = json.loads(canonical)
        if compare_keys is not None:
            existing_subset = {k: existing.get(k) for k in compare_keys}
            incoming_subset = {k: incoming.get(k) for k in compare_keys}
            if existing_subset == incoming_subset:
                return atomic_write_bytes(path, canonical)
            raise FileExistsError(f"Signed artifact differs on compare_keys: {path}")
        existing_stripped = canonical_json_bytes(_strip_provenance_metadata(existing))
        incoming_stripped = canonical_json_bytes(_strip_provenance_metadata(incoming))
        if existing_stripped != incoming_stripped:
            raise FileExistsError(f"Signed artifact differs from requested content: {path}")
        return path
    return atomic_write_bytes(path, canonical)


def write_text_once_or_verify(path: Path, value: str) -> Path:
    return write_bytes_once_or_verify(path, value.encode("utf-8"))


def write_text_once_or_verify_permissive(
    path: Path,
    value: str,
    compare_keys: list[str] | None = None,
) -> Path:
    """Write text atomically, or verify it if the file already exists.

    Permissive variant: if the file exists and the parsed CSV (or other
    structured text) differs, allow update when the ``compare_keys``
    whitelist matches.
    """
    content = value.encode("utf-8")
    if path.exists():
        existing = path.read_bytes()
        if existing == content:
            return path
        if compare_keys is not None:
            import csv
            import io
            try:
                existing_rows = list(csv.reader(io.StringIO(existing.decode("utf-8"))))
                incoming_rows = list(csv.reader(io.StringIO(value)))
                if len(existing_rows) > 0 and len(incoming_rows) > 0:
                    header = existing_rows[0]
                    key_indices = []
                    for ck in compare_keys:
                        if ck in header:
                            key_indices.append(header.index(ck))
                    def _subset(row):
                        return tuple(row[i] for i in key_indices) if key_indices else tuple(row)
                    existing_subset = sorted({_subset(r) for r in existing_rows[1:]})
                    incoming_subset = sorted({_subset(r) for r in incoming_rows[1:]})
                    if existing_subset == incoming_subset:
                        return atomic_write_bytes(path, content)
            except Exception:
                pass
            raise FileExistsError(f"Signed artifact text differs on compare_keys: {path}")
        raise FileExistsError(f"Signed artifact text differs from requested content: {path}")
    return atomic_write_bytes(path, content)


def csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
