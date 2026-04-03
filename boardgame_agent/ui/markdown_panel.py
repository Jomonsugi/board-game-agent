"""Markdown rendering helpers for non-PDF document citations.

Mirrors pdf_panel.py but renders markdown text with <mark> highlights
instead of PyMuPDF page images.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from boardgame_agent.config import DATA_DIR
from boardgame_agent.rag.extractor import load_cached_pages


def get_md_path(game_id: str, doc_name: str) -> Path | None:
    """Find the markdown file for a document, checking both docs/ and root."""
    for subdir in ("docs", ""):
        p = DATA_DIR / "games" / game_id / subdir / f"{doc_name}.md" if subdir else None
        if p and p.exists():
            return p
    p = DATA_DIR / "games" / game_id / "docs" / f"{doc_name}.md"
    return p if p.exists() else None


def render_highlighted_markdown(
    game_id: str,
    doc_name: str,
    page_num: int,
    bbox_indices: list[int],
) -> str | None:
    """Return HTML with cited bboxes wrapped in <mark> tags.

    Loads the cached extraction JSON, finds the section matching *page_num*,
    and highlights the text of the cited bboxes.
    """
    pages = load_cached_pages(game_id, doc_name)
    if pages is None:
        return None

    page_data = next((p for p in pages if p["page_num"] == page_num), None)
    if page_data is None:
        return None

    bboxes = page_data.get("bboxes", [])
    cited_texts = set()
    for idx in bbox_indices:
        if 0 <= idx < len(bboxes):
            cited_texts.add(bboxes[idx].get("text", ""))

    # Build the section text with highlights.
    section_text = page_data.get("text", "")
    if cited_texts:
        for ct in cited_texts:
            if ct and ct in section_text:
                section_text = section_text.replace(ct, f"<mark>{ct}</mark>")

    return section_text


def show_markdown_viewer(game_id: str, doc_name: str, scroll_to_section: int = 1) -> None:
    """Display the full markdown document in the Streamlit right panel."""
    md_path = get_md_path(game_id, doc_name)
    if md_path is None:
        st.warning(f"Markdown file not found: {doc_name}.md")
        return

    content = md_path.read_text(encoding="utf-8")
    st.markdown(content)
