"""Current-working-directory-independent project root discovery."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT_ENV = "PRACTICE_2_2_REPO_ROOT"


def _is_repo_root(candidate: Path) -> bool:
    return (
        (candidate / "total_practice" / "practice_2_2" / "README.md").is_file()
        and (candidate / "total_practice" / "practice_2").is_dir()
    )


def get_repo_root() -> Path:
    """Return the repository root without consulting the process cwd."""

    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not _is_repo_root(candidate):
            raise RuntimeError(
                f"{REPO_ROOT_ENV} does not identify this repository: {candidate}"
            )
        return candidate

    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        if _is_repo_root(candidate):
            return candidate
    raise RuntimeError(
        "Cannot discover repository root from the installed Practice 2.2 package; "
        f"set {REPO_ROOT_ENV} explicitly"
    )


def get_total_practice_root() -> Path:
    return get_repo_root() / "total_practice"


def get_practice_2_root() -> Path:
    path = get_total_practice_root() / "practice_2"
    if not path.is_dir():
        raise FileNotFoundError(f"Practice 2 compatibility root is missing: {path}")
    return path


def get_practice_2_2_root() -> Path:
    path = get_total_practice_root() / "practice_2_2"
    if not path.is_dir():
        raise FileNotFoundError(f"Practice 2.2 root is missing: {path}")
    return path
