"""Per-game icon dictionary: harvest → dedupe → resolve → apply.

Rulebook icons carry rule meaning ("this task must be completed second") that
no query-time model can recover from a caption-style description ("a red star
with a 2"). This module resolves icon meaning ONCE per game, offline, where a
frontier VLM and unlimited retries are affordable — then injects the resolved
meanings into the extraction cache so ordinary single-hop text RAG answers
icon questions.

The pipeline is on-demand (most games survive on text alone) and idempotent:
every stage caches its output, so re-running with a better model only touches
entries that aren't human-reviewed.

Stages
------
1. **harvest**  — crop every icon-sized raster placement in the game's PDFs
   (PyMuPDF, independent of Docling parse quality; composite pictures degrade
   gracefully because inline images are recorded individually). Blank crops
   (empty tracker boxes, whitespace) are skipped by pixel variance.
2. **dedupe**   — cluster instances by perceptual hash (dHash); recurring
   symbols collapse to one icon each, one-off art is dropped by a reuse
   threshold. Human-reviewed entries survive re-clustering.
3. **resolve**  — a configurable frontier VLM (``ICON_RESOLVE_MODEL``,
   Together/Anthropic/OpenAI) sees the icon crop PLUS the full candidate page
   and must quote the definition text, which is matched back to a bbox →
   ``definition citation`` (doc, page, bbox). Candidate pages are the pages
   where the icon actually appears, so legends, inline definitions, and
   dedicated reference documents are all handled by the same mechanism.
3b. **consolidate** — merge clusters that resolved to the same icon (same
   name, overlapping meaning). Hash clustering under-merges on purpose;
   meaning-level identity cleans up the duplicates safely. Auto-runs after
   resolve and before apply.
4. **apply**    — inject ``[Icon: name — meaning (defined in doc p.N)]``
   entries into the extraction cache as synthetic bboxes at the icon's actual
   position. Idempotent (previous injections are stripped first). Re-index
   afterwards for the meanings to reach search.

Storage: ``data/games/{game_id}/icons/icons.db`` (+ ``crops/*.png``).

CLI:
    python -m boardgame_agent.rag.icon_dictionary <game_id> [--stage all|harvest|dedupe|resolve|apply] [--model MODEL] [--force]
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import fitz  # PyMuPDF
from PIL import Image

fitz.TOOLS.mupdf_display_errors(False)

# ── Tunables (all overridable per call) ───────────────────────────────────────

# Icon-size bounds in PDF points (72pt = 1in). Icons that carry rule meaning
# sit inline with text or in margins, so print legibility conventions bound
# them: roughly caption-text height up to under an inch.
ICON_MAX_PTS = 60.0
ICON_MIN_PTS = 6.0

# dHash Hamming distance at or below which two crops are the same icon.
HASH_THRESHOLD = 6

# Clusters with fewer instances than this are one-off art, not recurring
# symbols (meaningful icons repeat because they encode the same rule).
MIN_INSTANCES = 3

# Grayscale pixel variance below which a crop is blank (empty form fields,
# tracker boxes, whitespace) and carries no icon.
BLANK_VARIANCE = 40.0

CROP_ZOOM = 4.0          # render scale for crops
MAX_CANDIDATE_PAGES = 3  # pages tried per icon during resolve
PAGE_RENDER_DPI = 150


# ── Paths & DB ────────────────────────────────────────────────────────────────

def _data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    from boardgame_agent.config import DATA_DIR
    return DATA_DIR


def icons_dir(game_id: str, data_dir: Path | None = None) -> Path:
    return _data_dir(data_dir) / "games" / game_id / "icons"


def db_path(game_id: str, data_dir: Path | None = None) -> Path:
    return icons_dir(game_id, data_dir) / "icons.db"


def has_dictionary(game_id: str, data_dir: Path | None = None) -> bool:
    return db_path(game_id, data_dir).exists()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS icons (
    icon_id      TEXT PRIMARY KEY,
    crop_path    TEXT NOT NULL,
    phash        TEXT NOT NULL,
    n_instances  INTEGER NOT NULL DEFAULT 0,
    name         TEXT,
    meaning      TEXT,
    status       TEXT NOT NULL DEFAULT 'new',
    def_doc      TEXT,
    def_page     INTEGER,
    def_bbox_idx INTEGER,
    def_quote    TEXT,
    model        TEXT,
    resolved_at  TEXT
);
CREATE TABLE IF NOT EXISTS icon_instances (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    icon_id        TEXT,
    doc_name       TEXT NOT NULL,
    pdf_page_index INTEGER NOT NULL,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    xref           INTEGER NOT NULL DEFAULT 0,
    phash          TEXT NOT NULL,
    crop_path      TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# Icon status lifecycle:
#   new        — harvested+deduped, not yet resolved
#   resolved   — VLM produced name/meaning WITH a matched definition citation
#   tentative  — VLM produced name/meaning but no verifiable definition
#   unresolved — VLM could not identify the icon
#   reviewed   — human approved/edited; never touched by re-runs
APPLIABLE_STATUSES = ("resolved", "tentative", "reviewed")


def connect(game_id: str, data_dir: Path | None = None) -> sqlite3.Connection:
    path = db_path(game_id, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


# ── Perceptual hashing ────────────────────────────────────────────────────────

def dhash(img: Image.Image) -> int:
    """64-bit difference hash: robust to scale/render differences, dependency-free."""
    g = img.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(g.tobytes())
    bits = 0
    for row in range(8):
        for col in range(8):
            left = px[row * 9 + col]
            right = px[row * 9 + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _is_blank(img: Image.Image, variance_threshold: float = BLANK_VARIANCE) -> bool:
    """True when a crop has no visual content (uniform fill / empty box)."""
    g = img.convert("L").resize((32, 32), Image.LANCZOS)
    px = list(g.tobytes())
    mean = sum(px) / len(px)
    var = sum((p - mean) ** 2 for p in px) / len(px)
    return var < variance_threshold


# ── Stage 1: harvest ──────────────────────────────────────────────────────────

def harvest(
    game_id: str,
    data_dir: Path | None = None,
    icon_max_pts: float = ICON_MAX_PTS,
    icon_min_pts: float = ICON_MIN_PTS,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Crop every icon-sized raster placement in the game's PDFs.

    Cached: a second call is a no-op unless *force*. Instances are recorded
    with their exact placement rect (PDF points, top-left origin) so
    citations can highlight the icon itself.
    """
    say = progress or (lambda _msg: None)
    ddir = _data_dir(data_dir)
    docs_dir = ddir / "games" / game_id / "docs"
    pdfs = sorted(docs_dir.glob("*.pdf")) if docs_dir.is_dir() else []
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found for game '{game_id}' at {docs_dir}")

    conn = connect(game_id, data_dir)
    try:
        n_existing = conn.execute("SELECT COUNT(*) FROM icon_instances").fetchone()[0]
        if n_existing and not force:
            return {"instances": n_existing, "skipped_blank": 0, "cached": True}
        conn.execute("DELETE FROM icon_instances")

        crops = icons_dir(game_id, data_dir) / "crops"
        crops.mkdir(parents=True, exist_ok=True)

        n_kept = 0
        n_blank = 0
        for pdf in pdfs:
            doc_name = pdf.stem
            say(f"Harvesting {doc_name} …")
            doc = fitz.open(str(pdf))
            try:
                for page in doc:
                    for i, info in enumerate(page.get_image_info(xrefs=True)):
                        r = fitz.Rect(info["bbox"]) & page.rect
                        if not (
                            icon_min_pts <= r.width <= icon_max_pts
                            and icon_min_pts <= r.height <= icon_max_pts
                        ):
                            continue
                        try:
                            pix = page.get_pixmap(
                                clip=r, matrix=fitz.Matrix(CROP_ZOOM, CROP_ZOOM)
                            )
                        except Exception:
                            continue
                        if pix.width < 4 or pix.height < 4:
                            continue
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        if _is_blank(img):
                            n_blank += 1
                            continue
                        h = dhash(img)
                        crop_path = crops / f"inst_{doc_name}_{page.number}_{i}.png"
                        img.save(crop_path)
                        conn.execute(
                            "INSERT INTO icon_instances "
                            "(doc_name, pdf_page_index, x0, y0, x1, y1, xref, phash, crop_path) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                doc_name, page.number,
                                r.x0, r.y0, r.x1, r.y1,
                                info.get("xref", 0) or 0,
                                f"{h:016x}",
                                str(crop_path.relative_to(ddir)),
                            ),
                        )
                        n_kept += 1
            finally:
                doc.close()

        _set_meta(conn, "harvested_at", datetime.now(timezone.utc).isoformat())
        conn.commit()
        say(f"Harvested {n_kept} icon-sized placements ({n_blank} blank crops skipped)")
        return {"instances": n_kept, "skipped_blank": n_blank, "cached": False}
    finally:
        conn.close()


# ── Stage 2: dedupe ───────────────────────────────────────────────────────────

def dedupe(
    game_id: str,
    data_dir: Path | None = None,
    hash_threshold: int = HASH_THRESHOLD,
    min_instances: int = MIN_INSTANCES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Cluster harvested instances into unique icons by perceptual hash.

    Preserves human work across re-runs: an existing ``icons`` row with the
    same icon_id keeps its name/meaning/status. Stale never-reviewed icons
    (no longer produced by clustering) are removed; reviewed ones are kept.
    """
    say = progress or (lambda _msg: None)
    conn = connect(game_id, data_dir)
    try:
        rows = conn.execute(
            "SELECT id, phash, crop_path, (x1-x0)*(y1-y0) AS area FROM icon_instances"
        ).fetchall()
        if not rows:
            raise RuntimeError("No harvested instances — run harvest first.")

        # Greedy clustering on Hamming distance to each cluster representative.
        clusters: list[dict[str, Any]] = []  # {rep: int, members: [row]}
        for row in rows:
            h = int(row["phash"], 16)
            best = None
            best_d = hash_threshold + 1
            for c in clusters:
                d = hamming(h, c["rep"])
                if d < best_d:
                    best, best_d = c, d
            if best is not None and best_d <= hash_threshold:
                best["members"].append(row)
            else:
                clusters.append({"rep": h, "members": [row]})

        kept = [c for c in clusters if len(c["members"]) >= min_instances]
        dropped = len(clusters) - len(kept)

        conn.execute("UPDATE icon_instances SET icon_id = NULL")
        new_ids: set[str] = set()
        for c in kept:
            canonical = max(c["members"], key=lambda r: r["area"])
            icon_id = f"icon_{int(canonical['phash'], 16):016x}"
            # Guard against hash collision across distinct clusters.
            while icon_id in new_ids:
                icon_id += "x"
            new_ids.add(icon_id)
            conn.execute(
                """INSERT INTO icons (icon_id, crop_path, phash, n_instances)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(icon_id) DO UPDATE SET
                     crop_path = excluded.crop_path,
                     phash = excluded.phash,
                     n_instances = excluded.n_instances""",
                (icon_id, canonical["crop_path"], canonical["phash"], len(c["members"])),
            )
            conn.executemany(
                "UPDATE icon_instances SET icon_id = ? WHERE id = ?",
                [(icon_id, r["id"]) for r in c["members"]],
            )

        # Drop stale icons that clustering no longer produces — unless reviewed.
        if new_ids:
            placeholders = ",".join("?" for _ in new_ids)
            conn.execute(
                f"DELETE FROM icons WHERE icon_id NOT IN ({placeholders}) "
                f"AND status != 'reviewed'",
                tuple(new_ids),
            )

        _set_meta(conn, "deduped_at", datetime.now(timezone.utc).isoformat())
        conn.commit()
        say(f"{len(kept)} unique icons ({dropped} one-off clusters dropped)")
        return {"icons": len(kept), "dropped_one_offs": dropped}
    finally:
        conn.close()


# ── Stage 3: resolve ──────────────────────────────────────────────────────────

_RESOLVE_PROMPT = """You are building an icon dictionary for the board game "{game_name}".

The FIRST image is a small icon cropped from the game's documents.
The SECOND image is a full page from the document "{doc_name}" where this icon appears. \
The page may be a legend/components section, a rules page that defines the icon inline, \
or a page that merely uses the icon without explaining it.

Answer in strict JSON (no markdown fences, no commentary):
{{
  "identified": true/false,        // could you determine what this icon means?
  "name": "short canonical name",  // e.g. "order token", "distress signal"
  "meaning": "one or two sentences: the rule this icon encodes, phrased so a player can act on it",
  "defined_here": true/false,      // does THIS page contain text that explains/defines the icon?
  "definition_quote": "exact contiguous words copied verbatim from this page that define the icon, or empty string"
}}

Rules:
- "meaning" must be the RULE, not a visual description. "A red star with a 2" is useless; \
"this task must be completed second" is correct.
- Only set defined_here=true if you can quote actual explanatory text from this page.
- The quote must be copied exactly so it can be located on the page."""


def _call_vlm(model: str, prompt: str, images: list[bytes]) -> str:
    """Call a vision-capable chat model with one or more PNG images."""
    import base64

    from boardgame_agent.config import ICON_VLM_OPTIONS, MODEL_OPTIONS

    provider = ICON_VLM_OPTIONS.get(model) or MODEL_OPTIONS.get(model, "together")
    b64s = [base64.standard_b64encode(png).decode() for png in images]

    if provider == "anthropic":
        import anthropic
        from boardgame_agent.config import ANTHROPIC_API_KEY
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        content: list[dict] = [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png", "data": b}}
            for b in b64s
        ]
        content.append({"type": "text", "text": prompt})
        resp = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text

    # Together and OpenAI both speak the OpenAI chat API.
    import openai
    if provider == "openai":
        from boardgame_agent.config import OPENAI_API_KEY
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        from boardgame_agent.config import TOGETHER_API_KEY
        client = openai.OpenAI(
            api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1"
        )
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}
        for b in b64s
    ]
    content.append({"type": "text", "text": prompt})
    resp = client.chat.completions.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content or ""


def _parse_vlm_json(text: str) -> dict[str, Any] | None:
    """Extract the JSON object from a VLM reply, tolerating markdown fences."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower()).strip()


def match_quote_to_bbox(page_data: dict[str, Any], quote: str) -> int | None:
    """Find the bbox index on a cached page whose text contains *quote*.

    Whitespace/punctuation-insensitive. Falls back to the bbox with the
    highest token overlap when no substring match exists (VLMs paraphrase
    slightly even when told not to). Returns None below a 60% overlap floor.
    """
    nq = _norm(quote)
    if not nq:
        return None
    q_tokens = set(nq.split())

    best_idx: int | None = None
    best_score = 0.0
    for idx, bbox in enumerate(page_data.get("bboxes", [])):
        nb = _norm(bbox.get("text") or "")
        if not nb:
            continue
        if nq in nb or nb in nq:
            return idx
        b_tokens = set(nb.split())
        if not q_tokens:
            continue
        score = len(q_tokens & b_tokens) / len(q_tokens)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx if best_score >= 0.6 else None


def _load_cache(
    game_id: str, doc_name: str, data_dir: Path | None = None
) -> list[dict[str, Any]] | None:
    """Load a cached extraction directly (no extractor import: that module
    pulls in Docling, and its loader ignores a data_dir override)."""
    p = _data_dir(data_dir) / "games" / game_id / "extracted" / f"{doc_name}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else None


def _logical_page_for_instance(
    pages_cache: list[dict[str, Any]],
    pdf_page_index: int,
    x_center: float,
    page_width: float,
) -> dict[str, Any] | None:
    """Map a physical placement to its logical (possibly spread-split) page."""
    candidates = [
        p for p in pages_cache
        if p.get("_pdf_page_index", p["page_num"] - 1) == pdf_page_index
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    half = "left" if x_center < page_width / 2 else "right"
    return next((p for p in candidates if p.get("_spread_half") == half), candidates[0])


def _render_logical_page(pdf_path: Path, page_data: dict[str, Any]) -> bytes | None:
    """Render a logical page (clipped to its spread half if needed) as PNG."""
    doc = fitz.open(str(pdf_path))
    try:
        idx = page_data.get("_pdf_page_index", page_data["page_num"] - 1)
        if idx >= doc.page_count:
            return None
        page = doc[idx]
        w, h = page.rect.width, page.rect.height
        half = page_data.get("_spread_half")
        if half == "left":
            clip = fitz.Rect(0, 0, w / 2, h)
        elif half == "right":
            clip = fitz.Rect(w / 2, 0, w, h)
        else:
            clip = page.rect
        pix = page.get_pixmap(dpi=PAGE_RENDER_DPI, clip=clip)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()


def _candidate_pages(
    conn: sqlite3.Connection,
    icon_id: str,
    game_id: str,
    data_dir: Path | None,
    doc_tags: dict[str, str],
    max_pages: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Rank the (doc, logical page) pairs where an icon appears.

    Definition pages carry explanatory text, so candidates are ranked by page
    text length, with a strong boost for rulebook-tagged documents. This is
    game-agnostic: legends, inline definitions, and dedicated icon-reference
    documents all surface because the icon appears on them.
    """
    ddir = _data_dir(data_dir)
    rows = conn.execute(
        "SELECT doc_name, pdf_page_index, (x0+x1)/2 AS xc FROM icon_instances "
        "WHERE icon_id = ?",
        (icon_id,),
    ).fetchall()

    caches: dict[str, list[dict[str, Any]] | None] = {}
    widths: dict[str, float] = {}
    scored: dict[tuple[str, int], tuple[float, dict[str, Any], str]] = {}
    for row in rows:
        doc_name = row["doc_name"]
        if doc_name not in caches:
            caches[doc_name] = _load_cache(game_id, doc_name, data_dir)
            pdf = ddir / "games" / game_id / "docs" / f"{doc_name}.pdf"
            if pdf.exists():
                d = fitz.open(str(pdf))
                widths[doc_name] = d[0].rect.width if d.page_count else 0.0
                d.close()
        pages = caches[doc_name]
        if not pages:
            continue
        page_data = _logical_page_for_instance(
            pages, row["pdf_page_index"], row["xc"], widths.get(doc_name, 0.0)
        )
        if page_data is None:
            continue
        key = (doc_name, page_data["page_num"])
        if key in scored:
            continue
        score = float(len(page_data.get("text") or ""))
        tag = doc_tags.get(doc_name, "rulebook")
        if tag in ("rulebook", "supplement", "quick_reference"):
            score += 100_000  # definitions live in rules material, not logbooks
        scored[key] = (score, page_data, doc_name)

    ranked = sorted(scored.values(), key=lambda t: t[0], reverse=True)
    return [(doc_name, page_data) for _, page_data, doc_name in ranked[:max_pages]]


def resolve(
    game_id: str,
    model: str | None = None,
    data_dir: Path | None = None,
    vlm_fn: Callable[[str, list[bytes]], str] | None = None,
    force: bool = False,
    max_pages: int = MAX_CANDIDATE_PAGES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resolve icon meanings with a frontier VLM, with definition citations.

    Only touches icons with status new/unresolved/tentative unless *force*
    (human-'reviewed' entries are never overwritten). *vlm_fn* overrides the
    API call — used by tests and available for local models.
    """
    say = progress or (lambda _msg: None)
    if model is None:
        from boardgame_agent.config import ICON_RESOLVE_MODEL
        model = ICON_RESOLVE_MODEL
    call = vlm_fn or (lambda prompt, images: _call_vlm(model, prompt, images))

    # Game name + doc tags for prompt/ranking (best effort; DB may be absent in tests).
    game_name = game_id
    doc_tags: dict[str, str] = {}
    try:
        from boardgame_agent.db.games import get_all_games, get_documents
        game_name = next(
            (g["game_name"] for g in get_all_games() if g["game_id"] == game_id), game_id
        )
        doc_tags = {
            d["doc_name"]: d.get("doc_tag", "rulebook") for d in get_documents(game_id)
        }
    except Exception:
        pass

    ddir = _data_dir(data_dir)
    conn = connect(game_id, data_dir)
    try:
        statuses = ("new", "unresolved", "tentative")
        if force:
            statuses = ("new", "unresolved", "tentative", "resolved")
        placeholders = ",".join("?" for _ in statuses)
        icons = conn.execute(
            f"SELECT * FROM icons WHERE status IN ({placeholders})", statuses
        ).fetchall()

        counts = {"resolved": 0, "tentative": 0, "unresolved": 0, "errors": 0}
        for icon in icons:
            crop_png = (ddir / icon["crop_path"]).read_bytes()
            say(f"Resolving {icon['icon_id']} ({icon['n_instances']} instances) …")

            result: dict[str, Any] = {"status": "unresolved"}
            for doc_name, page_data in _candidate_pages(
                conn, icon["icon_id"], game_id, data_dir, doc_tags, max_pages
            ):
                pdf = ddir / "games" / game_id / "docs" / f"{doc_name}.pdf"
                page_png = _render_logical_page(pdf, page_data) if pdf.exists() else None
                if page_png is None:
                    continue
                prompt = _RESOLVE_PROMPT.format(game_name=game_name, doc_name=doc_name)
                try:
                    reply = call(prompt, [crop_png, page_png])
                except Exception as e:  # noqa: BLE001 — one bad call must not kill the run
                    say(f"  VLM error on {doc_name} p.{page_data['page_num']}: {e}")
                    counts["errors"] += 1
                    continue
                data = _parse_vlm_json(reply)
                if not data or not data.get("identified"):
                    continue

                name = (data.get("name") or "").strip()
                meaning = (data.get("meaning") or "").strip()
                if not meaning:
                    continue
                if result["status"] == "unresolved":
                    result = {
                        "status": "tentative", "name": name, "meaning": meaning,
                        "def_doc": None, "def_page": None,
                        "def_bbox_idx": None, "def_quote": None,
                    }
                if data.get("defined_here") and data.get("definition_quote"):
                    bbox_idx = match_quote_to_bbox(page_data, data["definition_quote"])
                    if bbox_idx is not None:
                        result.update(
                            status="resolved", name=name, meaning=meaning,
                            def_doc=doc_name, def_page=page_data["page_num"],
                            def_bbox_idx=bbox_idx,
                            def_quote=data["definition_quote"],
                        )
                        break  # verified definition found — stop trying pages

            conn.execute(
                """UPDATE icons SET name = ?, meaning = ?, status = ?,
                   def_doc = ?, def_page = ?, def_bbox_idx = ?, def_quote = ?,
                   model = ?, resolved_at = ?
                   WHERE icon_id = ?""",
                (
                    result.get("name"), result.get("meaning"), result["status"],
                    result.get("def_doc"), result.get("def_page"),
                    result.get("def_bbox_idx"), result.get("def_quote"),
                    model, datetime.now(timezone.utc).isoformat(),
                    icon["icon_id"],
                ),
            )
            conn.commit()
            counts[result["status"]] += 1

        _set_meta(conn, "resolved_at", datetime.now(timezone.utc).isoformat())
        _set_meta(conn, "resolve_model", model)
        conn.commit()
        say(
            f"Resolve complete: {counts['resolved']} resolved, "
            f"{counts['tentative']} tentative, {counts['unresolved']} unresolved"
        )
        return counts
    finally:
        conn.close()


# ── Stage 3b: consolidate ─────────────────────────────────────────────────────

def consolidate(
    game_id: str,
    data_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Merge duplicate icons that resolved to the same thing.

    Perceptual-hash clustering is deliberately conservative (under-merging
    beats silently fusing two different rules), so the same glyph on
    different backgrounds can produce several clusters. After resolve, those
    duplicates are identifiable by MEANING rather than pixels: icons merge
    when their normalized names match AND their meanings substantially
    overlap (≥50% token overlap) — the second condition guards against a VLM
    giving one generic name (e.g. "chevron") to genuinely different icons.

    The surviving entry is chosen by status (reviewed > resolved > tentative),
    then by having a definition citation, then by instance count. Instances
    are repointed, so apply/highlighting cover every placement. Idempotent;
    runs automatically at the end of a dictionary build and before apply.
    """
    say = progress or (lambda _msg: None)
    rank = {"reviewed": 0, "resolved": 1, "tentative": 2}

    conn = connect(game_id, data_dir)
    try:
        placeholders = ",".join("?" for _ in APPLIABLE_STATUSES)
        rows = conn.execute(
            f"SELECT * FROM icons WHERE status IN ({placeholders}) "
            f"AND name IS NOT NULL AND name != '' "
            f"AND meaning IS NOT NULL AND meaning != ''",
            APPLIABLE_STATUSES,
        ).fetchall()

        by_name: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            by_name.setdefault(_norm(r["name"]), []).append(r)

        n_merged = 0
        for name, group in by_name.items():
            if len(group) < 2:
                continue
            group.sort(
                key=lambda r: (
                    rank.get(r["status"], 3),
                    0 if r["def_doc"] else 1,
                    -r["n_instances"],
                )
            )
            primary = group[0]
            p_tokens = set(_norm(primary["meaning"]).split())
            for dup in group[1:]:
                d_tokens = set(_norm(dup["meaning"]).split())
                overlap = len(p_tokens & d_tokens) / max(len(p_tokens | d_tokens), 1)
                if overlap < 0.5:
                    continue  # same name, different rule — keep both
                # Never fuse two human-reviewed entries; a human said both exist.
                if dup["status"] == "reviewed" and primary["status"] == "reviewed":
                    continue
                conn.execute(
                    "UPDATE icon_instances SET icon_id = ? WHERE icon_id = ?",
                    (primary["icon_id"], dup["icon_id"]),
                )
                conn.execute(
                    "UPDATE icons SET n_instances = n_instances + ? WHERE icon_id = ?",
                    (dup["n_instances"], primary["icon_id"]),
                )
                conn.execute("DELETE FROM icons WHERE icon_id = ?", (dup["icon_id"],))
                n_merged += 1
                say(f"Merged duplicate of '{primary['name']}' ({dup['icon_id']})")

        _set_meta(conn, "consolidated_at", datetime.now(timezone.utc).isoformat())
        conn.commit()
        return {"merged": n_merged}
    finally:
        conn.close()


# ── Stage 4: apply to extraction cache ────────────────────────────────────────

_ICON_BBOX_LABEL = "icon_meaning"


def format_icon_text(icon: sqlite3.Row | dict[str, Any]) -> str:
    """The text injected into the index for one icon instance."""
    get = icon.__getitem__ if isinstance(icon, sqlite3.Row) else icon.get
    text = f"[Icon: {get('name')} — {get('meaning')}]"
    if get("def_doc"):
        text = text[:-1] + f" (defined in {get('def_doc')} p.{get('def_page')})]"
    return text


def apply_to_cache(
    game_id: str,
    data_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Inject resolved icon meanings into the extraction cache, idempotently.

    For every logical page where a resolved icon appears, one synthetic bbox
    per icon is added at the first instance's actual position (label
    ``icon_meaning``), and its text is appended to the page text. Previous
    injections are stripped first, so re-applying after a better resolve or a
    human review never duplicates. Re-index the game afterwards.

    Returns {doc_name: n_injected_bboxes}.
    """
    say = progress or (lambda _msg: None)
    ddir = _data_dir(data_dir)
    consolidate(game_id, data_dir=data_dir, progress=progress)
    conn = connect(game_id, data_dir)
    try:
        placeholders = ",".join("?" for _ in APPLIABLE_STATUSES)
        icons = {
            r["icon_id"]: r
            for r in conn.execute(
                f"SELECT * FROM icons WHERE status IN ({placeholders}) "
                f"AND meaning IS NOT NULL AND meaning != ''",
                APPLIABLE_STATUSES,
            ).fetchall()
        }
        instances = conn.execute(
            "SELECT * FROM icon_instances WHERE icon_id IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    # Group instances by doc.
    by_doc: dict[str, list[sqlite3.Row]] = {}
    for inst in instances:
        if inst["icon_id"] in icons:
            by_doc.setdefault(inst["doc_name"], []).append(inst)

    report: dict[str, int] = {}
    extracted_dir = ddir / "games" / game_id / "extracted"
    for cache_path in sorted(extracted_dir.glob("*.json")):
        if cache_path.name.endswith(".images.json"):
            continue
        pages = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(pages, list):
            continue
        doc_name = cache_path.stem

        # 1) Strip previous injections (bboxes and their page-text lines).
        changed = False
        for page in pages:
            old = [b for b in page.get("bboxes", []) if b.get("label") == _ICON_BBOX_LABEL]
            if old:
                changed = True
                page["bboxes"] = [
                    b for b in page["bboxes"] if b.get("label") != _ICON_BBOX_LABEL
                ]
                text = page.get("text", "")
                for b in old:
                    text = text.replace("\n\n" + b["text"], "").replace(b["text"], "")
                page["text"] = text

        # 2) Inject current meanings.
        n_injected = 0
        doc_instances = by_doc.get(doc_name, [])
        if doc_instances:
            pdf = ddir / "games" / game_id / "docs" / f"{doc_name}.pdf"
            page_w = page_h = 0.0
            if pdf.exists():
                d = fitz.open(str(pdf))
                if d.page_count:
                    page_w, page_h = d[0].rect.width, d[0].rect.height
                d.close()

            seen: set[tuple[int, str]] = set()  # (logical page_num, icon_id)
            for inst in doc_instances:
                page_data = _logical_page_for_instance(
                    pages, inst["pdf_page_index"], (inst["x0"] + inst["x1"]) / 2, page_w
                )
                if page_data is None:
                    continue
                key = (page_data["page_num"], inst["icon_id"])
                if key in seen:
                    continue
                seen.add(key)
                icon = icons[inst["icon_id"]]
                text = format_icon_text(icon)

                # fitz top-left → Docling bottom-left (y0 = top edge, y0 > y1),
                # x shifted for right spread halves to match cached bboxes.
                x_off = page_w / 2 if page_data.get("_spread_half") == "right" else 0.0
                bbox = {
                    "x0": inst["x0"] - x_off,
                    "y0": page_h - inst["y0"],
                    "x1": inst["x1"] - x_off,
                    "y1": page_h - inst["y1"],
                    "text": text,
                    "label": _ICON_BBOX_LABEL,
                    "_icon_id": inst["icon_id"],
                }
                if icon["def_doc"]:
                    bbox["_definition"] = {
                        "doc_name": icon["def_doc"],
                        "page_num": icon["def_page"],
                        "bbox_idx": icon["def_bbox_idx"],
                    }
                page_data.setdefault("bboxes", []).append(bbox)
                page_data["text"] = (page_data.get("text", "") + "\n\n" + text).strip()
                n_injected += 1
                changed = True

        if changed:
            cache_path.write_text(json.dumps(pages), encoding="utf-8")
        if n_injected:
            say(f"{doc_name}: {n_injected} icon meaning(s) injected")
        report[doc_name] = n_injected
    return report


# ── Orchestration & stats ─────────────────────────────────────────────────────

def build_icon_dictionary(
    game_id: str,
    model: str | None = None,
    data_dir: Path | None = None,
    force: bool = False,
    vlm_fn: Callable[[str, list[bytes]], str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run harvest → dedupe → resolve. Apply/re-index remain explicit steps
    so a human can review the dictionary in between."""
    h = harvest(game_id, data_dir=data_dir, force=force, progress=progress)
    d = dedupe(game_id, data_dir=data_dir, progress=progress)
    r = resolve(
        game_id, model=model, data_dir=data_dir, force=force,
        vlm_fn=vlm_fn, progress=progress,
    )
    c = consolidate(game_id, data_dir=data_dir, progress=progress)
    return {"harvest": h, "dedupe": d, "resolve": r, "consolidate": c}


def get_stats(game_id: str, data_dir: Path | None = None) -> dict[str, Any] | None:
    """Summary for UI display; None when no dictionary exists."""
    if not has_dictionary(game_id, data_dir):
        return None
    conn = connect(game_id, data_dir)
    try:
        by_status = dict(
            conn.execute("SELECT status, COUNT(*) FROM icons GROUP BY status").fetchall()
        )
        n_instances = conn.execute(
            "SELECT COUNT(*) FROM icon_instances WHERE icon_id IS NOT NULL"
        ).fetchone()[0]
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        return {
            "icons": sum(by_status.values()),
            "by_status": by_status,
            "instances": n_instances,
            "meta": meta,
        }
    finally:
        conn.close()


def lookup(game_id: str, query: str, data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Keyword lookup over resolved icons (used by the agent's lookup_icon tool)."""
    if not has_dictionary(game_id, data_dir):
        return []
    conn = connect(game_id, data_dir)
    try:
        like = f"%{query.strip()}%"
        rows = conn.execute(
            "SELECT * FROM icons WHERE meaning IS NOT NULL AND meaning != '' "
            "AND (name LIKE ? OR meaning LIKE ?) ORDER BY n_instances DESC",
            (like, like),
        ).fetchall()
        if not rows and query.strip():
            # Fall back to listing everything — small tables, better than nothing.
            rows = conn.execute(
                "SELECT * FROM icons WHERE meaning IS NOT NULL AND meaning != '' "
                "ORDER BY n_instances DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a per-game icon dictionary.")
    parser.add_argument("game_id")
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "harvest", "dedupe", "resolve", "consolidate", "apply"],
    )
    parser.add_argument("--model", default=None, help="Vision model (see ICON_VLM_OPTIONS)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    say = print
    if args.stage == "all":
        report = build_icon_dictionary(
            args.game_id, model=args.model, force=args.force, progress=say
        )
        print(json.dumps(report, indent=2))
        print("\nReview the dictionary (UI or sqlite), then run --stage apply and re-index.")
    elif args.stage == "harvest":
        print(harvest(args.game_id, force=args.force, progress=say))
    elif args.stage == "dedupe":
        print(dedupe(args.game_id, progress=say))
    elif args.stage == "resolve":
        print(resolve(args.game_id, model=args.model, force=args.force, progress=say))
    elif args.stage == "consolidate":
        print(consolidate(args.game_id, progress=say))
    elif args.stage == "apply":
        report = apply_to_cache(args.game_id, progress=say)
        print(json.dumps(report, indent=2))
        print("Re-index the game for injected meanings to reach search.")


if __name__ == "__main__":
    _main()
