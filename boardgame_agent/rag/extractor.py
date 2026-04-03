"""Docling-based PDF extraction with JSON caching.

Docling reliably parses complex rulebook PDFs (multi-column, icons, tables)
and returns per-item provenance bounding boxes that power visual citations.

The output is cached as JSON so Docling only runs once per document.
Use force=True to re-extract (e.g. if the PDF is replaced).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    PictureDescriptionVlmEngineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from boardgame_agent.config import DATA_DIR, VLM_PICTURE_AREA_THRESHOLD

# Custom prompt tuned for board game rulebook icons and symbols.
_VLM_PROMPT = (
    "Describe what icons, symbols, numbers, or game components are shown "
    "in this image. Be concise (1-2 sentences)."
)


def _extract_single_pdf(
    pdf_path: Path, game_id: str, doc_name: str, vlm_preset: str | None = None
) -> list[dict[str, Any]]:
    """Run Docling on one PDF and return a list of per-page dicts.

    Each dict contains:
      - game_id, doc_name, page_num
      - text: full page text
      - bboxes: list of {x0, y0, x1, y1, text}  (Docling bottom-left origin, pts)

    When *vlm_preset* is set (e.g. ``"qwen"``), Docling's built-in VLM picture
    description is enabled.  Descriptions are written into the bbox ``text``
    field for picture items.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)

    if vlm_preset:
        pipeline_options.do_picture_description = True
        desc_options = PictureDescriptionVlmEngineOptions.from_preset(vlm_preset)
        desc_options.prompt = _VLM_PROMPT
        desc_options.picture_area_threshold = VLM_PICTURE_AREA_THRESHOLD
        pipeline_options.picture_description_options = desc_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path.resolve()))

    pages_data: list[dict[str, Any]] = []

    for page_num in sorted(result.document.pages.keys()):
        items_for_page = list(result.document.iterate_items(page_no=page_num))

        text_parts: list[str] = []
        bboxes: list[dict[str, Any]] = []

        for item, _ in items_for_page:
            label = str(item.label.value) if getattr(item, "label", None) else "text"
            item_text = ""

            if label == "table":
                # Tables have empty .text — use export_to_markdown() for structured content.
                if hasattr(item, "export_to_markdown"):
                    try:
                        item_text = item.export_to_markdown(doc=result.document)
                    except TypeError:
                        item_text = item.export_to_markdown()
            elif label == "picture":
                # Pictures: prefer VLM description if available, fall back to .text
                meta = getattr(item, "meta", None)
                if meta and getattr(meta, "description", None):
                    item_text = meta.description.text or ""
                elif getattr(item, "text", None):
                    item_text = str(item.text)
            elif getattr(item, "text", None):
                item_text = str(item.text)

            if item_text:
                text_parts.append(item_text)

            if getattr(item, "prov", None):
                for prov in item.prov:
                    if getattr(prov, "bbox", None):
                        bbox = prov.bbox
                        bbox_dict: dict[str, Any] = {
                            "x0": bbox.l,
                            "y0": bbox.t,
                            "x1": bbox.r,
                            "y1": bbox.b,
                            "text": item_text,
                            "label": label,
                        }
                        # Track which VLM model produced the description
                        if label == "picture" and item_text and vlm_preset:
                            bbox_dict["_vlm_model"] = vlm_preset
                        bboxes.append(bbox_dict)

        pages_data.append(
            {
                "game_id": game_id,
                "doc_name": doc_name,
                "page_num": page_num,
                "text": "\n\n".join(text_parts),
                "bboxes": bboxes,
            }
        )

    return pages_data


def _split_spreads(
    pages: list[dict[str, Any]],
    pdf_path: Path,
) -> list[dict[str, Any]]:
    """Split landscape spread pages into left/right virtual pages.

    A page is a spread if it is landscape (width > 1.2× height). Each spread becomes
    two virtual pages: bboxes are partitioned by the x-midpoint, and right-side
    bbox x-coordinates are shifted so they start at 0.

    Metadata added per page:
      - ``_pdf_page_index``: 0-based index into the physical PDF
      - ``_spread_half``: ``"left"`` | ``"right"`` | ``None``
    """
    import fitz

    doc = fitz.open(str(pdf_path.resolve()))
    try:
        result: list[dict[str, Any]] = []
        logical_page = 1

        for page_data in pages:
            pdf_idx = page_data["page_num"] - 1  # Docling is 1-indexed
            if pdf_idx >= doc.page_count:
                # Safety: if Docling gave a page beyond the PDF, pass through
                page_data["page_num"] = logical_page
                page_data["_pdf_page_index"] = pdf_idx
                page_data["_spread_half"] = None
                result.append(page_data)
                logical_page += 1
                continue

            fitz_page = doc[pdf_idx]
            w, h = fitz_page.rect.width, fitz_page.rect.height

            if w > 1.2 * h:
                # Spread page — split into left and right
                midpoint = w / 2.0
                bboxes = page_data.get("bboxes", [])

                left_bboxes: list[dict] = []
                right_bboxes: list[dict] = []
                left_texts: list[str] = []
                right_texts: list[str] = []

                for b in bboxes:
                    center_x = (b["x0"] + b["x1"]) / 2.0
                    if center_x < midpoint:
                        left_bboxes.append(b)
                        if b.get("text"):
                            left_texts.append(b["text"])
                    else:
                        # Shift x-coordinates so right half starts at 0
                        right_bboxes.append({
                            **b,
                            "x0": b["x0"] - midpoint,
                            "x1": b["x1"] - midpoint,
                        })
                        if b.get("text"):
                            right_texts.append(b["text"])

                # Left half
                result.append({
                    **page_data,
                    "page_num": logical_page,
                    "text": "\n\n".join(left_texts),
                    "bboxes": left_bboxes,
                    "_pdf_page_index": pdf_idx,
                    "_spread_half": "left",
                })
                logical_page += 1

                # Right half
                result.append({
                    **page_data,
                    "page_num": logical_page,
                    "text": "\n\n".join(right_texts),
                    "bboxes": right_bboxes,
                    "_pdf_page_index": pdf_idx,
                    "_spread_half": "right",
                })
                logical_page += 1
            else:
                # Single page — pass through with updated page_num
                page_data["page_num"] = logical_page
                page_data["_pdf_page_index"] = pdf_idx
                page_data["_spread_half"] = None
                result.append(page_data)
                logical_page += 1

        return result
    finally:
        doc.close()


def get_or_extract(
    doc_path: Path,
    game_id: str,
    doc_name: str,
    force: bool = False,
    has_spreads: bool = False,
    vlm_preset: str | None = None,
) -> list[dict[str, Any]]:
    """Return cached extraction for *doc_path*, running the appropriate extractor if needed.

    Dispatches based on file extension: .pdf → Docling, .md → markdown parser.
    Cache lives at data/games/{game_id}/extracted/{doc_name}.json.
    Pass force=True to ignore the cache and re-extract.
    Pass has_spreads=True to split landscape spread pages into two logical pages.
    Pass vlm_preset (e.g. ``"qwen"``) to enable VLM picture descriptions.
    """
    cache_path = DATA_DIR / "games" / game_id / "extracted" / f"{doc_name}.json"

    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    ext = doc_path.suffix.lower()
    if ext == ".md":
        from boardgame_agent.rag.markdown_extractor import extract_markdown
        print(f"  Parsing markdown: {doc_path.name} …")
        pages = extract_markdown(doc_path, game_id, doc_name)
    else:
        print(f"  Docling extracting: {doc_path.name} …")
        pages = _extract_single_pdf(doc_path, game_id, doc_name, vlm_preset=vlm_preset)
        if has_spreads:
            pages = _split_spreads(pages, doc_path)
            n_spreads = sum(1 for p in pages if p.get("_spread_half") == "left")
            if n_spreads:
                print(f"  Split {n_spreads} spread page(s) → {len(pages)} logical pages")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def re_enrich_pictures(
    game_id: str,
    doc_name: str,
    vlm_preset: str,
    has_spreads: bool = False,
) -> int:
    """Re-extract a document with VLM picture descriptions enabled.

    This re-runs the full Docling extraction (Docling's VLM is a pipeline
    stage, not a standalone post-processor). The cached JSON is overwritten.
    Returns the number of picture bboxes that received VLM descriptions.
    """
    from boardgame_agent.ui.pdf_panel import get_pdf_path

    pdf_path = get_pdf_path(game_id, doc_name)
    if pdf_path is None:
        raise FileNotFoundError(f"PDF not found for {doc_name}")

    print(f"  Re-enriching {doc_name} with VLM preset '{vlm_preset}' …")
    pages = get_or_extract(
        pdf_path, game_id, doc_name,
        force=True,
        has_spreads=has_spreads,
        vlm_preset=vlm_preset,
    )

    # Count how many picture bboxes got VLM descriptions
    count = sum(
        1
        for page in pages
        for b in page.get("bboxes", [])
        if b.get("label") == "picture" and b.get("_vlm_model")
    )
    print(f"  {doc_name}: {count} pictures enriched with {vlm_preset}")
    return count


def load_cached_pages(game_id: str, doc_name: str) -> list[dict[str, Any]] | None:
    """Load already-cached Docling output without running extraction."""
    cache_path = DATA_DIR / "games" / game_id / "extracted" / f"{doc_name}.json"
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


_HEADING_LABELS = {"section_header", "title"}
_TABLE_LABEL = "table"


def chunk_by_sections(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split page-level dicts into section-level chunks using bbox labels.

    Chunking rules:
    - Each table bbox becomes its own isolated chunk (tables are complete units).
    - Non-table bboxes are grouped by heading: a new chunk starts at each
      section_header/title. Pages with no headings emit as a single chunk.
    - Lone-heading runs (heading with no body) are merged into the next run.

    Returns chunk dicts with ``original_bbox_indices`` mapping back to the
    original page bbox list (used by the retriever for citation display).
    """
    chunks: list[dict[str, Any]] = []

    def _emit(bbox_indices: list[int], page: dict[str, Any], bboxes: list[dict[str, Any]]) -> None:
        chunk_bboxes = [bboxes[j] for j in bbox_indices]
        chunk_text = "\n\n".join(b["text"] for b in chunk_bboxes if b.get("text"))
        if not chunk_text.strip():
            return
        chunk = {
            "game_id": page["game_id"],
            "doc_name": page["doc_name"],
            "page_num": page["page_num"],
            "text": chunk_text,
            "bboxes": chunk_bboxes,
            "original_bbox_indices": bbox_indices,
        }
        # Carry through any extra metadata (e.g. doc_tag) from the page dict.
        if "doc_tag" in page:
            chunk["doc_tag"] = page["doc_tag"]
        chunks.append(chunk)

    for page in pages:
        bboxes: list[dict[str, Any]] = page.get("bboxes", [])
        if not bboxes:
            continue

        # Separate tables out first — each becomes its own chunk immediately.
        # Collect non-table indices for section-based chunking below.
        non_table_indices: list[int] = []
        for idx, bbox in enumerate(bboxes):
            if bbox.get("label", "text") == _TABLE_LABEL:
                _emit([idx], page, bboxes)
            else:
                non_table_indices.append(idx)

        if not non_table_indices:
            continue

        # Group non-table bboxes into runs: start a new run at each heading.
        runs: list[list[int]] = []
        current: list[int] = []
        for idx in non_table_indices:
            label = bboxes[idx].get("label", "text")
            if label in _HEADING_LABELS and current:
                runs.append(current)
                current = [idx]
            else:
                current.append(idx)
        if current:
            runs.append(current)

        # Merge lone-heading runs into the following run.
        merged: list[list[int]] = []
        i = 0
        while i < len(runs):
            run = runs[i]
            if (
                len(run) == 1
                and bboxes[run[0]].get("label", "text") in _HEADING_LABELS
                and i + 1 < len(runs)
            ):
                merged.append(run + runs[i + 1])
                i += 2
            else:
                merged.append(run)
                i += 1

        for bbox_indices in merged:
            _emit(bbox_indices, page, bboxes)

    return chunks


def extract_source(
    source: Path,
    game_id: str,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Extract from a single PDF or every PDF in a folder.

    Returns all pages across all documents, each tagged with doc_name.
    """
    source = Path(source)
    pdf_paths = sorted(source.glob("*.pdf")) if source.is_dir() else [source]
    if not pdf_paths:
        raise ValueError(f"No PDF files found at {source}")

    all_pages: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        doc_name = pdf_path.stem
        all_pages.extend(get_or_extract(pdf_path, game_id, doc_name, force=force))
    return all_pages
