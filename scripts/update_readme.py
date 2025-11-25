#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"
NOTES_DIR = ROOT / "notes"

START_TAG = "<!-- NOTES_START -->"
END_TAG = "<!-- NOTES_END -->"


def get_note_title(path: Path) -> str:

    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except (UnicodeDecodeError, IndexError):
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            first_line = file.readline().strip()
    else:
        first_line = first_line.strip()

    if first_line.startswith("#"):
        return first_line.lstrip("#").strip() or path.stem
    return first_line or path.stem


def build_notes_list() -> str:

    if not NOTES_DIR.exists():
        return "_暂时还没有笔记_"

    md_files = sorted(
        NOTES_DIR.glob("*.md"),
        key=lambda p: p.name,
        reverse=True,
    )

    if not md_files:
        return "_暂时还没有笔记_"

    lines = []
    for path in md_files:
        title = get_note_title(path)
        rel_path = path.relative_to(ROOT).as_posix()
        lines.append(f"- [{title}]({rel_path})")

    return "\n".join(lines)


def replace_block(content: str, new_block: str) -> str:
    if START_TAG not in content or END_TAG not in content:
        raise RuntimeError("README.md 里缺少 NOTES_START 或 NOTES_END 标记")

    before, _middle = content.split(START_TAG, 1)
    _old_block, after = _middle.split(END_TAG, 1)
    return before + START_TAG + "\n\n" + new_block + "\n\n" + END_TAG + after


def update_readme() -> None:
    notes_md = build_notes_list()
    content = README_PATH.read_text(encoding="utf-8")
    updated = replace_block(content, notes_md)
    README_PATH.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    update_readme()
