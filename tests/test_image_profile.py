"""Tests for the programmatic icon-detection strategy decision."""

from boardgame_agent.rag.probe_pdf_images import decide_icon_strategy


def _report(**overrides):
    base = {
        "icon_candidates": [{"xref": 1, "count": 5, "w_pts": 23, "h_pts": 23, "pages": [1]}],
        "n_raster_placements": 100,
        "n_inline_images": 0,
        "avg_drawings_per_page": 5.0,
    }
    base.update(overrides)
    return base


def test_clean_xref_world():
    assert decide_icon_strategy(_report()) == "xref"


def test_inline_heavy_needs_hybrid():
    # Candidates exist, but most placements are inline images (xref 0),
    # which xref identity cannot track — common in print-production PDFs.
    assert decide_icon_strategy(_report(n_inline_images=75)) == "hybrid"


def test_vector_heavy_needs_hybrid():
    assert decide_icon_strategy(_report(avg_drawings_per_page=168.8)) == "hybrid"


def test_no_candidates_means_render():
    assert decide_icon_strategy(
        _report(icon_candidates=[], avg_drawings_per_page=200.0)
    ) == "render"


def test_no_rasters_at_all():
    assert decide_icon_strategy(
        _report(icon_candidates=[], n_raster_placements=0, n_inline_images=0)
    ) == "render"
