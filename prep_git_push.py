"""One-off: extend .gitignore for large CSV/Excel; add placeholders in empty dirs."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 99 * 1024 * 1024  # GitHub hard limit 100 MiB
DATA_EXTS = {".csv", ".xlsx", ".xls", ".xlsm"}
PLACEHOLDER = "_empty_dir_placeholder.txt"
SKIP_LOG = "GIT_PUSH_SKIPPED_LARGE_FILES.txt"
MARK_BEGIN = "# BEGIN LARGE FILE SKIPS (auto prep_git_push.py)"
MARK_END = "# END LARGE FILE SKIPS"


def should_skip_walk_dir(name: str) -> bool:
    return name in {".venv", ".git"}


def large_skip_paths() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not should_skip_walk_dir(d)]
        rel_base = os.path.relpath(dirpath, ROOT)
        if rel_base.startswith(".venv") or rel_base.startswith(".git"):
            continue
        for fn in filenames:
            if fn == "prep_git_push.py":
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in DATA_EXTS:
                continue
            fp = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            if sz > MAX_BYTES:
                rel = os.path.relpath(fp, ROOT).replace("\\", "/")
                out.append((rel, sz))
    out.sort(key=lambda x: -x[1])
    return out


def patch_gitignore(paths: list[str]) -> None:
    gi = os.path.join(ROOT, ".gitignore")
    body = ""
    if os.path.isfile(gi):
        with open(gi, encoding="utf-8") as f:
            body = f.read()
    if MARK_BEGIN in body and MARK_END in body:
        i0 = body.index(MARK_BEGIN)
        i1 = body.index(MARK_END) + len(MARK_END)
        body = (body[:i0] + body[i1:]).strip()
    block_lines = [MARK_BEGIN, ""] + ["/" + p for p in paths] + ([""] if paths else []) + [MARK_END]
    new_body = body + ("\n\n" if body else "") + "\n".join(block_lines) + "\n"
    with open(gi, "w", encoding="utf-8") as f:
        f.write(new_body)


def write_skip_log(items: list[tuple[str, int]]) -> None:
    lines = [
        "Not committed: CSV/Excel over ~99 MiB (GitHub file size limit).",
        "Shrink, use Git LFS, or host elsewhere; then remove paths from .gitignore block.",
        "",
    ]
    for p, sz in items:
        lines.append(f"{p}\t{sz} bytes")
    lines.append("")
    with open(os.path.join(ROOT, SKIP_LOG), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def add_empty_placeholders() -> int:
    created = 0
    for dirpath, dirnames, filenames in os.walk(ROOT, topdown=False):
        if should_skip_walk_dir(os.path.basename(dirpath)):
            continue
        rel = os.path.relpath(dirpath, ROOT)
        if rel.startswith(".venv") or rel.startswith(".git"):
            continue
        real_files = [f for f in filenames if f != PLACEHOLDER]
        if real_files:
            continue
        ph = os.path.join(dirpath, PLACEHOLDER)
        if not os.path.isfile(ph):
            with open(ph, "w", encoding="utf-8") as f:
                f.write("Placeholder so Git tracks this otherwise-empty directory.\n")
            created += 1
    return created


def main() -> int:
    items = large_skip_paths()
    patch_gitignore([p for p, _ in items])
    write_skip_log(items)
    n = add_empty_placeholders()
    print(f"Skipped large data files: {len(items)} -> {SKIP_LOG}")
    print(f"Empty-dir placeholders added: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
