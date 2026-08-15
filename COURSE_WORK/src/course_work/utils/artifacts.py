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


def write_json_once_or_verify(path: Path, value: Any) -> Path:
    return write_bytes_once_or_verify(path, canonical_json_bytes(value))


def write_text_once_or_verify(path: Path, value: str) -> Path:
    return write_bytes_once_or_verify(path, value.encode("utf-8"))


def csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
