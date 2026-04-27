"""view_page tool — visual page analysis via VLM."""

from __future__ import annotations

import base64
import io
from typing import Any

import fitz  # PyMuPDF

fitz.TOOLS.mupdf_display_errors(False)

from PIL import Image
from langchain_core.tools import tool

from boardgame_agent.config import DATA_DIR, PAGE_VISION_DPI, PAGE_VISION_MODEL


def _render_page_png(game_id: str, doc_name: str, page_data: dict[str, Any]) -> bytes | None:
    """Render a full page as PNG bytes for sending to a VLM API."""
    pdf_path = DATA_DIR / "games" / game_id / "docs" / f"{doc_name}.pdf"
    if not pdf_path.exists():
        return None

    doc = fitz.open(str(pdf_path.resolve()))
    try:
        pdf_idx = page_data.get("_pdf_page_index", page_data["page_num"] - 1)
        if pdf_idx >= doc.page_count:
            return None
        fitz_page = doc[pdf_idx]
        spread_half = page_data.get("_spread_half")
        page_width = fitz_page.rect.width
        page_height = fitz_page.rect.height

        if spread_half == "left":
            clip = fitz.Rect(0, 0, page_width / 2, page_height)
        elif spread_half == "right":
            clip = fitz.Rect(page_width / 2, 0, page_width, page_height)
        else:
            clip = fitz_page.rect

        pix = fitz_page.get_pixmap(dpi=PAGE_VISION_DPI, clip=clip)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()


def _call_vlm(prompt: str, image_png: bytes, provider: str) -> str:
    """Call a vision-capable LLM with an image and text prompt."""
    img_b64 = base64.standard_b64encode(image_png).decode()

    if provider == "anthropic":
        import anthropic
        from boardgame_agent.config import ANTHROPIC_API_KEY
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=PAGE_VISION_MODEL, max_tokens=2048,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return response.content[0].text

    elif provider == "openai":
        import openai
        from boardgame_agent.config import OPENAI_API_KEY
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=PAGE_VISION_MODEL, max_tokens=2048,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return response.choices[0].message.content

    elif provider == "together":
        import openai
        from boardgame_agent.config import TOGETHER_API_KEY
        client = openai.OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
        response = client.chat.completions.create(
            model=PAGE_VISION_MODEL, max_tokens=2048,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return response.choices[0].message.content

    raise ValueError(f"Unsupported VLM provider: {provider}")


def make_page_vision_tool(game_id: str):
    """Return a view_page tool bound to *game_id*."""

    @tool
    def view_page(doc_name: str, page_num: int, question: str) -> str:
        """Visually analyze a page to understand its layout, icons, or visual content.

        Use this when you found a page via search but can't understand it
        from the extracted text alone (e.g. icon-heavy pages, visual layouts).
        The result helps you understand what to search for next — it does NOT
        replace searching for the actual rules. Always follow up by searching
        for the terms and concepts the vision analysis reveals.

        Args:
            doc_name: The document name (as shown in search results).
            page_num: The page number to view.
            question: What you want to understand about this page.
        """
        from boardgame_agent.config import MODEL_OPTIONS
        from boardgame_agent.rag.extractor import load_cached_pages

        pages = load_cached_pages(game_id, doc_name)
        if pages is None:
            return f"Document '{doc_name}' not found or not yet extracted."

        page_data = next((p for p in pages if p["page_num"] == page_num), None)
        if page_data is None:
            return f"Page {page_num} not found in '{doc_name}'."

        page_png = _render_page_png(game_id, doc_name, page_data)
        if page_png is None:
            return f"Could not render page {page_num} of '{doc_name}'."

        provider = MODEL_OPTIONS.get(PAGE_VISION_MODEL, "anthropic")

        prompt = (
            f"You are analyzing page {page_num} of a board game rulebook "
            f"document called '{doc_name}'.\n\n"
            f"Question: {question}\n\n"
            f"Describe what you see that answers this question. Be specific "
            f"about any icons, symbols, numbers, or game components visible."
        )

        try:
            return _call_vlm(prompt, page_png, provider)
        except Exception as e:
            return f"Vision analysis failed: {e}"

    return view_page
