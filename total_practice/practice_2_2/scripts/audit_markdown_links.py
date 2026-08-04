"""Fail on broken local Markdown links in active Practice 2/2.2 documentation."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
PROJECTS = (ROOT / "practice_2", ROOT / "practice_2_2")
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def main() -> None:
    broken = []
    checked = 0
    for project in PROJECTS:
        for document in project.rglob("*.md"):
            if "archive" in document.parts or "migration" in document.parts:
                continue
            for raw in LINK.findall(document.read_text(errors="replace")):
                target = raw.strip().split()[0].strip("<>")
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_text = unquote(target.split("#", 1)[0])
                if not path_text:
                    continue
                checked += 1
                if not (document.parent / path_text).resolve().exists():
                    broken.append(f"{document.relative_to(ROOT)} -> {target}")
    if broken:
        raise RuntimeError("Broken local Markdown links:\n" + "\n".join(broken))
    print(f"MARKDOWN_LINK_AUDIT_PASS checked={checked}")


if __name__ == "__main__":
    main()
