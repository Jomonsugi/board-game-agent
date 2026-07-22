"""Probe a PDF's embedded images to assess icon-detection strategies.

Rulebook PDFs place icons in one of two ways:

1. **Raster XObjects** — the same embedded image resource (identified by its
   xref) is reused at every placement. If icons are rasters, ``xref identity``
   gives exact icon deduplication and placement rects for free, with no
   dependence on Docling's picture parsing.
2. **Vector drawings** — icons drawn as paths don't appear as image XObjects.
   Detection then needs rendering-based approaches (template matching over
   rendered pages, or Docling picture bboxes).

The probe runs AUTOMATICALLY at ingestion time: ``get_or_extract`` profiles
every PDF and caches the result as ``extracted/{doc_name}.images.json``,
including a machine-readable ``icon_strategy`` field ("xref" | "hybrid" |
"render") that downstream consumers (e.g. an icon-dictionary builder) read
instead of asking the user. Nothing is decided manually.

The CLI remains for ad-hoc inspection of PDFs that aren't ingested yet:

    python -m boardgame_agent.rag.probe_pdf_images <pdf-or-dir> [...]

Output per document:
  - embedded raster images: unique xrefs, reuse histogram, sizes
  - likely-icon candidates (small, reused rasters)
  - vector drawing count per page (proxy for vector-drawn iconography)
  - the decided icon_strategy
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import fitz

# Default classification thresholds. Both are overridable per call.
#
# ICON_MAX_PTS: placements at or below this size (PDF points; 72pt = 1 inch)
# count as icon-sized. Icons that carry rule meaning are set inline with text
# or in margins, so they're bounded by print legibility conventions: roughly
# caption-text height (~0.25") up to under an inch. Anything larger is
# typically an illustration, diagram, or photo rather than a symbol.
ICON_MAX_PTS = 60.0
# ICON_MIN_REUSE: an image placed this many times or more is likely a
# recurring symbol rather than a one-off illustration. Meaningful icons repeat
# because they encode the same rule wherever they appear; decorative art
# rarely repeats at small size.
ICON_MIN_REUSE = 3


def probe_pdf(
    pdf_path: Path,
    icon_max_pts: float = ICON_MAX_PTS,
    icon_min_reuse: int = ICON_MIN_REUSE,
) -> dict:
    """Return a report dict for one PDF (see module docstring)."""
    doc = fitz.open(str(pdf_path))
    try:
        placements: list[dict] = []   # every raster placement on every page
        drawings_per_page: list[int] = []

        for page in doc:
            for info in page.get_image_info(xrefs=True):
                r = fitz.Rect(info["bbox"])
                placements.append(
                    {
                        "page": page.number + 1,
                        "xref": info["xref"],
                        "w_pts": r.width,
                        "h_pts": r.height,
                    }
                )
            drawings_per_page.append(len(page.get_drawings()))

        reuse = Counter(p["xref"] for p in placements if p["xref"] > 0)
        # xref 0 = inline image (not an XObject); identity tracking impossible.
        n_inline = sum(1 for p in placements if p["xref"] == 0)

        icon_candidates = []
        for xref, count in reuse.most_common():
            sizes = [
                (p["w_pts"], p["h_pts"]) for p in placements if p["xref"] == xref
            ]
            w, h = sizes[0]
            if count >= icon_min_reuse and w <= icon_max_pts and h <= icon_max_pts:
                pages = sorted({p["page"] for p in placements if p["xref"] == xref})
                icon_candidates.append(
                    {"xref": xref, "count": count, "w_pts": round(w, 1),
                     "h_pts": round(h, 1), "pages": pages}
                )

        return {
            "pdf": pdf_path.name,
            "n_pages": doc.page_count,
            "n_raster_placements": len(placements),
            "n_unique_xrefs": len(reuse),
            "n_inline_images": n_inline,
            "n_reused_xrefs": sum(1 for c in reuse.values() if c > 1),
            "icon_candidates": icon_candidates,
            "total_vector_drawings": sum(drawings_per_page),
            "avg_drawings_per_page": round(
                sum(drawings_per_page) / max(len(drawings_per_page), 1), 1
            ),
        }
    finally:
        doc.close()


def decide_icon_strategy(report: dict) -> str:
    """Decide, programmatically, how icons should be detected in this PDF.

    Returns one of:
      - ``"xref"``   — reused raster icons exist and nearly all raster
                       placements are trackable XObjects: xref identity alone
                       identifies icons and their placements.
      - ``"hybrid"`` — trackable icon rasters exist, but a meaningful share of
                       placements are inline images (xref 0) or the page is
                       vector-heavy: use xref identity where available and
                       rendering-based matching (perceptual hash / template
                       matching over rendered pages) for the rest.
      - ``"render"`` — no reused icon-sized rasters: icons are vector-drawn or
                       baked into larger images; only rendering-based
                       approaches will find them.
    """
    has_candidates = bool(report["icon_candidates"])
    n = max(report["n_raster_placements"], 1)
    untrackable_share = report["n_inline_images"] / n
    vector_heavy = report["avg_drawings_per_page"] > 50

    if not has_candidates:
        return "render"
    if untrackable_share > 0.2 or vector_heavy:
        return "hybrid"
    return "xref"


def profile_pdf(
    pdf_path: Path,
    icon_max_pts: float = ICON_MAX_PTS,
    icon_min_reuse: int = ICON_MIN_REUSE,
) -> dict:
    """Full ingestion-time profile: probe report + decided icon strategy."""
    report = probe_pdf(pdf_path, icon_max_pts=icon_max_pts, icon_min_reuse=icon_min_reuse)
    report["icon_strategy"] = decide_icon_strategy(report)
    return report


def format_report(r: dict) -> str:
    lines = [
        f"── {r['pdf']} ({r['n_pages']} pages) " + "─" * 20,
        f"  raster placements : {r['n_raster_placements']} "
        f"({r['n_unique_xrefs']} unique xrefs, {r['n_reused_xrefs']} reused, "
        f"{r['n_inline_images']} inline/untrackable)",
        f"  vector drawings   : {r['total_vector_drawings']} total "
        f"({r['avg_drawings_per_page']}/page)",
    ]
    if r["icon_candidates"]:
        lines.append(f"  icon-sized reused rasters ({len(r['icon_candidates'])}):")
        for c in r["icon_candidates"][:15]:
            pages = ", ".join(map(str, c["pages"][:8]))
            more = "…" if len(c["pages"]) > 8 else ""
            lines.append(
                f"    xref {c['xref']:>4}  ×{c['count']:<3} "
                f"{c['w_pts']:.0f}×{c['h_pts']:.0f}pt  pages {pages}{more}"
            )
        if len(r["icon_candidates"]) > 15:
            lines.append(f"    … and {len(r['icon_candidates']) - 15} more")
    else:
        lines.append("  icon-sized reused rasters: NONE")

    strategy = r.get("icon_strategy") or decide_icon_strategy(r)
    explanations = {
        "xref": "reused raster icons, nearly all trackable — xref identity suffices",
        "hybrid": "trackable icon rasters plus inline/vector content — xref identity where available, rendering-based matching for the rest",
        "render": "no reused icon-sized rasters — icons are vector-drawn or baked into larger images; rendering-based matching required",
    }
    lines.append(f"  icon_strategy: {strategy} ({explanations[strategy]})")
    return "\n".join(lines)


def _main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    pdfs: list[Path] = []
    for a in args:
        p = Path(a)
        pdfs.extend(sorted(p.glob("**/*.pdf")) if p.is_dir() else [p])
    if not pdfs:
        print("No PDFs found.")
        sys.exit(1)

    for pdf in pdfs:
        print(format_report(profile_pdf(pdf)))
        print()


if __name__ == "__main__":
    _main()
