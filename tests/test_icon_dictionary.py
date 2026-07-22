"""Tests for the icon-dictionary pipeline (harvest → dedupe → resolve → apply).

Fixtures are fully synthetic: a PDF is built with PyMuPDF containing two
distinct recurring "icons" (high-contrast deterministic patterns), a one-off
image, a large illustration, and a blank box — exercising size filtering,
blank-crop skipping, clustering, and the reuse threshold without depending on
any real game. The VLM is mocked via the resolve stage's ``vlm_fn`` hook.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw

from boardgame_agent.rag.icon_dictionary import (
    apply_to_cache,
    connect,
    consolidate,
    dedupe,
    dhash,
    format_icon_text,
    hamming,
    harvest,
    lookup,
    match_quote_to_bbox,
    resolve,
    _is_blank,
    _load_cache,
    _logical_page_for_instance,
)

GAME = "testgame"


# ── Image fixtures ────────────────────────────────────────────────────────────

def _icon_a() -> Image.Image:
    """Vertical split: left black, right white."""
    img = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(img).rectangle([0, 0, 31, 63], fill="black")
    return img


def _icon_b() -> Image.Image:
    """8×8 checkerboard."""
    img = Image.new("RGB", (64, 64), "white")
    d = ImageDraw.Draw(img)
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 0:
                d.rectangle([c * 8, r * 8, c * 8 + 7, r * 8 + 7], fill="black")
    return img


def _icon_c() -> Image.Image:
    """Horizontal split: top black, bottom white (one-off)."""
    img = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(img).rectangle([0, 0, 63, 31], fill="black")
    return img


def _png(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── PDF + cache fixture ───────────────────────────────────────────────────────

@pytest.fixture()
def game_dir(tmp_path: Path) -> Path:
    """Build data/games/testgame/{docs,extracted} with a synthetic rulebook."""
    docs = tmp_path / "games" / GAME / "docs"
    extracted = tmp_path / "games" / GAME / "extracted"
    docs.mkdir(parents=True)
    extracted.mkdir(parents=True)

    a, b, c = _png(_icon_a()), _png(_icon_b()), _png(_icon_c())
    blank = _png(Image.new("RGB", (64, 64), (200, 200, 200)))

    doc = fitz.open()
    pages_cache = []
    for pno in range(3):
        page = doc.new_page(width=400, height=600)
        # Icon A at a fixed spot on every page (3 instances).
        page.insert_image(fitz.Rect(50, 50, 80, 80), stream=a)
        # Icon B on every page (3 instances).
        page.insert_image(fitz.Rect(120, 50, 150, 80), stream=b)
        if pno == 0:
            # One-off icon C, a large illustration, and a blank box.
            page.insert_image(fitz.Rect(200, 50, 230, 80), stream=c)
            page.insert_image(fitz.Rect(50, 200, 350, 500), stream=a)  # too big
            page.insert_image(fitz.Rect(300, 50, 330, 80), stream=blank)

        bboxes = [
            {"x0": 40, "y0": 560, "x1": 360, "y1": 520,
             "text": "Some rules prose for this page.", "label": "text"},
        ]
        if pno == 1:
            bboxes.append({
                "x0": 40, "y0": 500, "x1": 360, "y1": 460,
                "text": "The split icon means you must complete this task second.",
                "label": "text",
            })
        pages_cache.append({
            "game_id": GAME, "doc_name": "rules", "page_num": pno + 1,
            "text": "\n\n".join(x["text"] for x in bboxes),
            "bboxes": bboxes,
        })
    doc.save(docs / "rules.pdf")
    doc.close()
    (extracted / "rules.json").write_text(json.dumps(pages_cache))
    return tmp_path


# ── Hashing primitives ────────────────────────────────────────────────────────

def test_dhash_identical_and_distinct():
    assert hamming(dhash(_icon_a()), dhash(_icon_a().resize((32, 32)))) <= 2
    assert hamming(dhash(_icon_a()), dhash(_icon_b())) > 6
    assert hamming(dhash(_icon_a()), dhash(_icon_c())) > 6


def test_blank_detection():
    assert _is_blank(Image.new("RGB", (64, 64), (180, 180, 180)))
    assert not _is_blank(_icon_b())


# ── Harvest + dedupe ──────────────────────────────────────────────────────────

def test_harvest_filters_and_caches(game_dir: Path):
    report = harvest(GAME, data_dir=game_dir)
    # 3×A + 3×B + 1×C kept; large illustration filtered by size; blank skipped.
    assert report["instances"] == 7
    assert report["skipped_blank"] == 1
    # Idempotent: second call is a cache hit.
    again = harvest(GAME, data_dir=game_dir)
    assert again["cached"] is True and again["instances"] == 7


def test_dedupe_clusters_and_drops_one_offs(game_dir: Path):
    harvest(GAME, data_dir=game_dir)
    report = dedupe(GAME, data_dir=game_dir)
    assert report["icons"] == 2           # A and B recur; C is a one-off
    assert report["dropped_one_offs"] == 1

    conn = connect(GAME, data_dir=game_dir)
    counts = dict(conn.execute(
        "SELECT icon_id, COUNT(*) FROM icon_instances "
        "WHERE icon_id IS NOT NULL GROUP BY icon_id"
    ).fetchall())
    conn.close()
    assert sorted(counts.values()) == [3, 3]


def test_dedupe_preserves_reviewed_entries(game_dir: Path):
    harvest(GAME, data_dir=game_dir)
    dedupe(GAME, data_dir=game_dir)
    conn = connect(GAME, data_dir=game_dir)
    iid = conn.execute("SELECT icon_id FROM icons LIMIT 1").fetchone()[0]
    conn.execute(
        "UPDATE icons SET name='order token', meaning='human meaning', "
        "status='reviewed' WHERE icon_id = ?", (iid,),
    )
    conn.commit()
    conn.close()

    dedupe(GAME, data_dir=game_dir)  # re-run must not clobber the review
    conn = connect(GAME, data_dir=game_dir)
    row = conn.execute("SELECT * FROM icons WHERE icon_id = ?", (iid,)).fetchone()
    conn.close()
    assert row["status"] == "reviewed" and row["meaning"] == "human meaning"


# ── Quote → bbox matching ─────────────────────────────────────────────────────

def test_match_quote_exact_and_fuzzy():
    page = {"bboxes": [
        {"text": "Unrelated prose about spaceships."},
        {"text": "The split icon means you must complete this task second."},
    ]}
    assert match_quote_to_bbox(page, "you MUST complete this task second") == 1
    # Light paraphrase still lands on the right bbox via token overlap.
    assert match_quote_to_bbox(page, "split icon means complete the task second") == 1
    assert match_quote_to_bbox(page, "totally different words entirely") is None
    assert match_quote_to_bbox(page, "") is None


# ── Logical page mapping (spreads) ────────────────────────────────────────────

def test_logical_page_mapping_spreads():
    pages = [
        {"page_num": 1, "_pdf_page_index": 0, "_spread_half": "left"},
        {"page_num": 2, "_pdf_page_index": 0, "_spread_half": "right"},
        {"page_num": 3, "_pdf_page_index": 1, "_spread_half": None},
    ]
    assert _logical_page_for_instance(pages, 0, 100, 800)["page_num"] == 1
    assert _logical_page_for_instance(pages, 0, 700, 800)["page_num"] == 2
    assert _logical_page_for_instance(pages, 1, 100, 800)["page_num"] == 3
    assert _logical_page_for_instance(pages, 9, 100, 800) is None


# ── Resolve (mocked VLM) ──────────────────────────────────────────────────────

def _mock_vlm_factory(calls: list):
    """VLM that identifies everything and quotes the page-2 definition."""
    def vlm(prompt: str, images: list[bytes]) -> str:
        calls.append(prompt)
        return json.dumps({
            "identified": True,
            "name": "split icon",
            "meaning": "You must complete this task second.",
            "defined_here": True,
            "definition_quote": "you must complete this task second",
        })
    return vlm


def test_resolve_with_citation(game_dir: Path):
    harvest(GAME, data_dir=game_dir)
    dedupe(GAME, data_dir=game_dir)
    calls: list = []
    report = resolve(GAME, model="mock", data_dir=game_dir, vlm_fn=_mock_vlm_factory(calls))
    assert report["resolved"] == 2 and report["unresolved"] == 0
    assert calls  # the VLM was actually consulted

    conn = connect(GAME, data_dir=game_dir)
    rows = conn.execute("SELECT * FROM icons").fetchall()
    conn.close()
    for row in rows:
        assert row["status"] == "resolved"
        assert row["def_doc"] == "rules"
        assert row["def_page"] == 2          # the quote only matches page 2's bbox
        assert row["def_bbox_idx"] == 1
        assert row["model"] == "mock"


def test_resolve_unidentified_is_unresolved_and_rerunnable(game_dir: Path):
    harvest(GAME, data_dir=game_dir)
    dedupe(GAME, data_dir=game_dir)
    refuse = lambda p, i: json.dumps({"identified": False})
    report = resolve(GAME, model="weak", data_dir=game_dir, vlm_fn=refuse)
    assert report["unresolved"] == 2

    # A better model later picks up exactly the unresolved entries.
    calls: list = []
    report2 = resolve(GAME, model="strong", data_dir=game_dir, vlm_fn=_mock_vlm_factory(calls))
    assert report2["resolved"] == 2


def test_resolve_never_touches_reviewed(game_dir: Path):
    harvest(GAME, data_dir=game_dir)
    dedupe(GAME, data_dir=game_dir)
    conn = connect(GAME, data_dir=game_dir)
    conn.execute("UPDATE icons SET status='reviewed', name='human', meaning='human'")
    conn.commit()
    conn.close()

    calls: list = []
    resolve(GAME, model="mock", data_dir=game_dir, vlm_fn=_mock_vlm_factory(calls), force=True)
    assert calls == []  # nothing to resolve → no VLM calls


# ── Apply ─────────────────────────────────────────────────────────────────────

def _distinct_vlm(prompt: str, images: list[bytes]) -> str:
    """Mock VLM giving each icon a distinct name (keyed by crop bytes),
    so consolidation inside apply_to_cache leaves both icons intact."""
    import hashlib
    tag = hashlib.sha1(images[0]).hexdigest()[:6]
    return json.dumps({
        "identified": True,
        "name": f"split icon {tag}",
        "meaning": "You must complete this task second.",
        "defined_here": True,
        "definition_quote": "you must complete this task second",
    })


def test_apply_injects_and_is_idempotent(game_dir: Path):
    harvest(GAME, data_dir=game_dir)
    dedupe(GAME, data_dir=game_dir)
    resolve(GAME, model="mock", data_dir=game_dir, vlm_fn=_distinct_vlm)

    report = apply_to_cache(GAME, data_dir=game_dir)
    assert report["rules"] == 6  # 2 icons × 3 pages

    pages = _load_cache(GAME, "rules", data_dir=game_dir)
    first_snapshot = json.dumps(pages, sort_keys=True)
    injected = [b for p in pages for b in p["bboxes"] if b["label"] == "icon_meaning"]
    assert len(injected) == 6
    for b in injected:
        assert "[Icon: split icon" in b["text"]
        assert "(defined in rules p.2)" in b["text"]
        assert b["_definition"]["page_num"] == 2
        # Docling bottom-left convention: y0 (top) > y1 (bottom).
        assert b["y0"] > b["y1"]
    # Meaning reached the page text (that's what gets indexed).
    assert "[Icon: split icon" in pages[0]["text"]

    # Re-apply must not duplicate anything.
    report2 = apply_to_cache(GAME, data_dir=game_dir)
    assert report2["rules"] == 6
    pages2 = _load_cache(GAME, "rules", data_dir=game_dir)
    assert json.dumps(pages2, sort_keys=True) == first_snapshot


# ── Consolidate ───────────────────────────────────────────────────────────────

def _seed_icon(conn, icon_id, name, meaning, status="resolved",
               def_doc=None, n_instances=3):
    conn.execute(
        "INSERT INTO icons (icon_id, crop_path, phash, n_instances, name, "
        "meaning, status, def_doc, def_page) VALUES (?, 'x.png', '0', ?, ?, ?, ?, ?, 2)",
        (icon_id, n_instances, name, meaning, status, def_doc),
    )
    for i in range(n_instances):
        conn.execute(
            "INSERT INTO icon_instances (icon_id, doc_name, pdf_page_index, "
            "x0, y0, x1, y1, xref, phash) VALUES (?, 'rules', ?, 0, 0, 9, 9, 0, '0')",
            (icon_id, i),
        )


def test_consolidate_merges_same_name_and_meaning(tmp_path: Path):
    conn = connect(GAME, data_dir=tmp_path)
    _seed_icon(conn, "icon_a", "order token 1", "Complete this task first.",
               def_doc="rules", n_instances=5)
    _seed_icon(conn, "icon_b", "Order Token 1", "You must complete this task first.",
               n_instances=3)
    _seed_icon(conn, "icon_c", "distress signal", "Rotate cards and pass one.",
               n_instances=4)
    conn.commit()
    conn.close()

    report = consolidate(GAME, data_dir=tmp_path)
    assert report["merged"] == 1

    conn = connect(GAME, data_dir=tmp_path)
    rows = {r["icon_id"]: r for r in conn.execute("SELECT * FROM icons").fetchall()}
    # icon_a survives (has citation), absorbing icon_b's instances.
    assert set(rows) == {"icon_a", "icon_c"}
    assert rows["icon_a"]["n_instances"] == 8
    n_repointed = conn.execute(
        "SELECT COUNT(*) FROM icon_instances WHERE icon_id = 'icon_a'"
    ).fetchone()[0]
    conn.close()
    assert n_repointed == 8

    # Idempotent.
    assert consolidate(GAME, data_dir=tmp_path)["merged"] == 0


def test_consolidate_guards(tmp_path: Path):
    conn = connect(GAME, data_dir=tmp_path)
    # Same generic name, different rules — must NOT merge.
    _seed_icon(conn, "icon_a", "chevron", "Communicate your highest card.")
    _seed_icon(conn, "icon_b", "chevron", "The commander leads the first trick.")
    # Two reviewed entries with same name+meaning — human said both exist; keep.
    _seed_icon(conn, "icon_c", "omega", "Complete this task last.", status="reviewed")
    _seed_icon(conn, "icon_d", "omega", "Complete this task last.", status="reviewed")
    conn.commit()
    conn.close()

    report = consolidate(GAME, data_dir=tmp_path)
    assert report["merged"] == 0

    conn = connect(GAME, data_dir=tmp_path)
    n = conn.execute("SELECT COUNT(*) FROM icons").fetchone()[0]
    conn.close()
    assert n == 4


def test_consolidate_prefers_reviewed_as_primary(tmp_path: Path):
    conn = connect(GAME, data_dir=tmp_path)
    _seed_icon(conn, "icon_auto", "rocket", "Win the trick with the rocket.",
               def_doc="rules", n_instances=10)
    _seed_icon(conn, "icon_human", "rocket", "Win this trick with the rocket card.",
               status="reviewed", n_instances=3)
    conn.commit()
    conn.close()

    consolidate(GAME, data_dir=tmp_path)
    conn = connect(GAME, data_dir=tmp_path)
    rows = conn.execute("SELECT * FROM icons").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["icon_id"] == "icon_human"  # reviewed wins over cited+popular
    assert rows[0]["n_instances"] == 13


def test_build_pipeline_consolidates_duplicates_end_to_end(game_dir: Path):
    harvest(GAME, data_dir=game_dir)
    dedupe(GAME, data_dir=game_dir)
    # Mock VLM gives BOTH icons the same name+meaning → consolidate to one.
    resolve(GAME, model="mock", data_dir=game_dir, vlm_fn=_mock_vlm_factory([]))
    report = consolidate(GAME, data_dir=game_dir)
    assert report["merged"] == 1

    # Apply now injects one meaning per page instead of two.
    apply_report = apply_to_cache(GAME, data_dir=game_dir)
    assert apply_report["rules"] == 3  # 1 icon × 3 pages
    pages = _load_cache(GAME, "rules", data_dir=game_dir)
    injected = [b for p in pages for b in p["bboxes"] if b["label"] == "icon_meaning"]
    assert len(injected) == 3


# ── Lookup + formatting ───────────────────────────────────────────────────────

def test_lookup_and_format(game_dir: Path):
    harvest(GAME, data_dir=game_dir)
    dedupe(GAME, data_dir=game_dir)
    resolve(GAME, model="mock", data_dir=game_dir, vlm_fn=_mock_vlm_factory([]))

    matches = lookup(GAME, "split", data_dir=game_dir)
    assert len(matches) == 2
    text = format_icon_text(matches[0])
    assert text.startswith("[Icon: split icon — You must complete this task second.")
    assert "(defined in rules p.2)" in text

    # Unmatched query falls back to listing all resolved icons.
    assert len(lookup(GAME, "zzz-no-match", data_dir=game_dir)) == 2
    # No dictionary → empty.
    assert lookup("no_such_game", "x") == []
