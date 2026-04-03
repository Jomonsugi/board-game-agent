"""Markdown extraction — parse .md files into the same page-dict format as Docling PDFs.

Each top-level heading (# or ##) becomes a logical "page" (section). Within
each section, paragraphs and sub-headings become bboxes with appropriate labels
so chunk_by_sections() can split them the same way it splits PDF content.

Bbox coordinates are zeroed (no visual layout) — they're only used for PyMuPDF
rendering which doesn't apply to markdown. The text content in each bbox is
what matters for citation highlighting in the markdown viewer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)


def extract_markdown(
    md_path: Path,
    game_id: str,
    doc_name: str,
) -> list[dict[str, Any]]:
    """Parse a markdown file into a list of per-section page dicts.

    Each section starts at a # or ## heading. Sub-headings (### and below)
    are treated as content within the section, not section boundaries.

    Returns the same structure as _extract_single_pdf() in extractor.py:
      - game_id, doc_name, page_num, text, bboxes
    """
    text = md_path.read_text(encoding="utf-8")

    # Find all top-level heading positions.
    headings = list(_HEADING_RE.finditer(text))

    if not headings:
        # No headings — treat the entire file as one section.
        return [_make_page(text, game_id, doc_name, page_num=1)]

    pages: list[dict[str, Any]] = []

    # Content before the first heading (if any).
    preamble = text[: headings[0].start()].strip()
    if preamble:
        pages.append(_make_page(preamble, game_id, doc_name, page_num=len(pages) + 1))

    for i, match in enumerate(headings):
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            pages.append(_make_page(section_text, game_id, doc_name, page_num=len(pages) + 1))

    return pages


def _make_page(
    section_text: str,
    game_id: str,
    doc_name: str,
    page_num: int,
) -> dict[str, Any]:
    """Build a page dict from a markdown section, creating bboxes from paragraphs."""
    blocks = _split_into_blocks(section_text)
    bboxes: list[dict[str, Any]] = []

    for block in blocks:
        label = "section_header" if block.startswith("#") else "text"
        # Strip heading markers for the text content.
        clean = re.sub(r"^#+\s+", "", block).strip() if label == "section_header" else block
        bboxes.append({
            "x0": 0, "y0": 0, "x1": 0, "y1": 0,
            "text": clean,
            "label": label,
        })

    return {
        "game_id": game_id,
        "doc_name": doc_name,
        "page_num": page_num,
        "text": section_text,
        "bboxes": bboxes,
    }


def _split_into_blocks(text: str) -> list[str]:
    """Split markdown text into blocks (headings and paragraphs).

    Consecutive non-blank lines form a single block. Blank lines separate blocks.
    Headings (lines starting with #) always start a new block.
    """
    blocks: list[str] = []
    current: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        if not stripped:
            # Blank line — flush current block.
            if current:
                blocks.append("\n".join(current))
                current = []
            continue

        if stripped.startswith("#") and current:
            # Heading starts a new block.
            blocks.append("\n".join(current))
            current = [line]
            continue

        current.append(line)

    if current:
        blocks.append("\n".join(current))

    return blocks
