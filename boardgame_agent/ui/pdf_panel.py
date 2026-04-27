"""PDF rendering helpers.

Provides two complementary views:
  1. render_highlighted_page — PyMuPDF renders a single page with bbox highlights
     as a PIL Image. Used when the user clicks a citation.
  2. show_pdf_viewer — renders the full scrollable PDF using streamlit-pdf-viewer.
     Used as the persistent right-panel PDF browser.

Supports spread-split pages: when a landscape spread was split during extraction,
``_pdf_page_index`` and ``_spread_half`` in the cached page data control which
physical half-page to render and how to map bbox coordinates.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from boardgame_agent.config import DATA_DIR
from boardgame_agent.rag.extractor import load_cached_pages


def get_pdf_path(game_id: str, doc_name: str) -> Path | None:
    p = DATA_DIR / "games" / game_id / "docs" / f"{doc_name}.pdf"
    return p if p.exists() else None


def render_highlighted_page(
    game_id: str,
    doc_name: str,
    page_num: int,
    bbox_indices: list[int],
    dpi: int = 150,
) -> Image.Image | None:
    """Render *page_num* of *doc_name* with cited bboxes highlighted in yellow.

    Handles spread-split pages: uses ``_pdf_page_index`` to find the physical
    PDF page and ``_spread_half`` to clip to the correct half. Bbox coordinates
    in the cache are already adjusted for the half-page.
    """
    pdf_path = get_pdf_path(game_id, doc_name)
    if pdf_path is None:
        return None

    pages = load_cached_pages(game_id, doc_name)
    if pages is None:
        return None

    page_data = next((p for p in pages if p["page_num"] == page_num), None)
    if page_data is None:
        return None

    bboxes = page_data.get("bboxes", [])
    pdf_page_index = page_data.get("_pdf_page_index", page_num - 1)
    spread_half = page_data.get("_spread_half")

    doc = fitz.open(str(pdf_path.resolve()))
    try:
        if pdf_page_index >= doc.page_count:
            return None
        fitz_page = doc[pdf_page_index]
        page_width = fitz_page.rect.width
        page_height = fitz_page.rect.height

        # For spread pages, determine the clip rect for the correct half
        if spread_half == "left":
            clip = fitz.Rect(0, 0, page_width / 2, page_height)
            x_offset = 0.0
        elif spread_half == "right":
            clip = fitz.Rect(page_width / 2, 0, page_width, page_height)
            # Bboxes were shifted to start at 0 during extraction,
            # so we need to shift them back for rendering on the full page
            x_offset = page_width / 2
        else:
            clip = fitz_page.rect
            x_offset = 0.0

        # The effective page height for coordinate conversion is the same
        # (spreads share the same height).
        effective_height = page_height

        for idx in bbox_indices:
            if 0 <= idx < len(bboxes):
                b = bboxes[idx]
                x0, y0, x1, y1 = b["x0"] + x_offset, b["y0"], b["x1"] + x_offset, b["y1"]
                # Docling: bottom-left origin → PyMuPDF: top-left origin
                top_y0 = effective_height - y1
                top_y1 = effective_height - y0
                rect = fitz.Rect(min(x0, x1), min(top_y0, top_y1), max(x0, x1), max(top_y0, top_y1))
                annot = fitz_page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1, 1, 0))
                annot.update()

        pix = fitz_page.get_pixmap(dpi=dpi, clip=clip)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def show_pdf_viewer(game_id: str, doc_name: str, scroll_to_page: int = 1) -> None:
    """Display the full scrollable PDF in the Streamlit right panel.

    For spread-split documents, renders each half-page as a separate image
    instead of using the PDF viewer (which would show full spreads).
    """
    import streamlit as st

    pdf_path = get_pdf_path(game_id, doc_name)
    if pdf_path is None:
        st.warning(f"PDF not found: {doc_name}.pdf")
        return

    pages = load_cached_pages(game_id, doc_name)
    has_spreads = pages and any(p.get("_spread_half") for p in pages)

    if has_spreads:
        # Render individual half-pages as images for spread documents
        if pages is None:
            return
        for page_data in pages:
            pnum = page_data["page_num"]
            img = render_highlighted_page(game_id, doc_name, pnum, [])
            if img:
                st.image(img, caption=f"Page {pnum}")
    else:
        try:
            from streamlit_pdf_viewer import pdf_viewer

            pdf_viewer(
                input=str(pdf_path),
                height=700,
                scroll_to_page=scroll_to_page,
            )
        except ImportError:
            st.info(
                "Install `streamlit-pdf-viewer` for the embedded viewer. "
                f"Currently showing: **{doc_name}** · Page {scroll_to_page}"
            )
            img = render_highlighted_page(game_id, doc_name, scroll_to_page, [])
            if img:
                st.image(img, caption=f"{doc_name} · Page {scroll_to_page}")
