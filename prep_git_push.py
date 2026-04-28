"""Prep for git push: placeholders in dirs that only contain ignored spreadsheet files."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER = "_empty_dir_placeholder.txt"
# Matching .gitignore: dirs with only these (plus placeholders) still need a placeholder.
DATA_EXTS = {".csv", ".xlsx", ".xls", ".xlsm", ".xlsb"}


def should_skip_walk_dir(name: str) -> bool:
    return name in {".venv", ".git"}


def is_ignored_tabular(fn: str) -> bool:
    ext = os.path.splitext(fn)[1].lower()
    return ext in DATA_EXTS


def add_empty_placeholders() -> int:
    created = 0
    for dirpath, dirnames, filenames in os.walk(ROOT, topdown=False):
        if should_skip_walk_dir(os.path.basename(dirpath)):
            continue
        rel = os.path.relpath(dirpath, ROOT)
        if rel.startswith(".venv") or rel.startswith(".git"):
            continue
        # Files Git should track besides placeholder — ignore spreadsheet-only dirs
        tractable = [
            f
            for f in filenames
            if f != PLACEHOLDER and not is_ignored_tabular(f)
        ]
        if tractable:
            continue
        ph = os.path.join(dirpath, PLACEHOLDER)
        if not os.path.isfile(ph):
            with open(ph, "w", encoding="utf-8") as f:
                f.write("Placeholder so Git tracks this otherwise-empty directory.\n")
            created += 1
    return created


def main() -> int:
    n = add_empty_placeholders()
    print(f"Empty-dir placeholders added: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
