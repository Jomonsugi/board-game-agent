"""Sanitize VLM picture descriptions before they reach the index.

Local VLMs fail in two recurring ways that poison retrieval:

1. **Refusals** — near-blank or low-content crops (empty tracker boxes, blank
   form fields, decorative whitespace regions) make the model respond with
   "I'm sorry, but I cannot see any image attached to your question."
   Indexed verbatim, these strings match nothing useful and dilute search.
2. **Chat-template leakage** — generation runs past the answer into special
   tokens, e.g. "...a red star with the number 2 inside it.<|im_end".
   The partial token noise pollutes embeddings.

Both failure modes are model-generic (any chat-tuned VLM can produce them),
not game- or rulebook-specific. This module is dependency-free so it can be
imported and tested in isolation.

Two entry points:

- ``sanitize_vlm_description`` — used at extraction time (extractor.py).
- ``retro_sanitize_game`` / CLI — cleans already-cached extraction JSON in
  place without re-running Docling, so improving these rules never means
  starting extraction from scratch. Re-indexing IS required afterwards
  (the dirty strings are already embedded in Qdrant).

CLI:
    python -m boardgame_agent.rag.sanitize <game_id>      # one game
    python -m boardgame_agent.rag.sanitize --all          # every game
    python -m boardgame_agent.rag.sanitize <game_id> --dry-run
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Everything from the first special-token opener onward is template leakage.
# Covers ChatML (<|im_end|>), Llama (<|eot_id|>), partial cutoffs ("<|im_"), etc.
_SPECIAL_TOKEN_RE = re.compile(r"<\|.*$", re.DOTALL)

# Refusal / no-image phrases. A description containing any of these carries no
# visual information and must not be indexed. Case-insensitive, model-generic.
_REFUSAL_PATTERNS = [
    r"\bcannot see any image\b",
    r"\bcan(?:no|')t see (?:the|an|any) image\b",
    r"\bno image (?:is )?(?:provided|attached|present|visible)\b",
    r"\bthere is no image\b",
    r"\bplease provide (?:an|the) image\b",
    r"\bunable to (?:see|view|describe) (?:the|an|any) image\b",
    r"\bcannot describe the image\b",
    r"\bi'?m sorry,? but i cannot\b",
    r"\bas an ai\b.{0,40}\bcannot\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# Below this length a "description" is noise ("A.", "img"), not signal.
_MIN_USEFUL_LENGTH = 12


def sanitize_vlm_description(text: str | None) -> str:
    """Return a cleaned VLM description, or "" if it carries no information.

    Empty return value means "treat this picture as undescribed": the bbox
    keeps its coordinates (citations still work) but contributes no text to
    the page, so it is invisible to search instead of poisoning it.
    """
    if not text:
        return ""
    cleaned = _SPECIAL_TOKEN_RE.sub("", text).strip()
    if len(cleaned) < _MIN_USEFUL_LENGTH:
        return ""
    if _REFUSAL_RE.search(cleaned):
        return ""
    return cleaned


def sanitize_page(page: dict[str, Any]) -> int:
    """Sanitize all picture-bbox texts on a cached page dict, in place.

    Rewrites both the bbox ``text`` fields and the occurrences of the dirty
    strings inside the page-level ``text`` (which was concatenated from item
    texts at extraction time). Returns the number of bboxes changed.
    """
    changed = 0
    page_text: str = page.get("text", "")

    for bbox in page.get("bboxes", []):
        if bbox.get("label") != "picture":
            continue
        original = bbox.get("text") or ""
        cleaned = sanitize_vlm_description(original)
        if cleaned == original:
            continue
        bbox["text"] = cleaned
        changed += 1
        if original and original in page_text:
            replacement = cleaned if cleaned else ""
            page_text = page_text.replace(original, replacement)

    if changed:
        # Collapse blank lines left behind by removed descriptions.
        page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
        page["text"] = page_text
    return changed


def sanitize_pages(pages: list[dict[str, Any]]) -> int:
    """Sanitize a full cached extraction (list of page dicts), in place."""
    return sum(sanitize_page(p) for p in pages)


# ── Retro-clean cached extractions ───────────────────────────────────────────

def retro_sanitize_game(
    game_id: str, data_dir: Path | None = None, dry_run: bool = False
) -> dict[str, int]:
    """Clean every cached extraction JSON for *game_id* in place.

    Returns ``{doc_name: n_bboxes_cleaned}``. Documents with zero dirty
    bboxes are left untouched. Callers must re-index afterwards for the
    cleanup to reach Qdrant.
    """
    if data_dir is None:
        from boardgame_agent.config import DATA_DIR
        data_dir = DATA_DIR

    extracted_dir = data_dir / "games" / game_id / "extracted"
    if not extracted_dir.is_dir():
        raise FileNotFoundError(f"No extracted cache for game '{game_id}' at {extracted_dir}")

    report: dict[str, int] = {}
    for cache_path in sorted(extracted_dir.glob("*.json")):
        pages = json.loads(cache_path.read_text(encoding="utf-8"))
        n = sanitize_pages(pages)
        report[cache_path.stem] = n
        if n and not dry_run:
            cache_path.write_text(json.dumps(pages), encoding="utf-8")
    return report


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Retro-clean VLM descriptions in cached extractions.")
    parser.add_argument("game_id", nargs="?", help="Game ID (folder name under data/games/)")
    parser.add_argument("--all", action="store_true", help="Process every game")
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    args = parser.parse_args()

    from boardgame_agent.config import DATA_DIR

    if args.all:
        game_ids = sorted(p.name for p in (DATA_DIR / "games").iterdir() if p.is_dir())
    elif args.game_id:
        game_ids = [args.game_id]
    else:
        parser.error("Provide a game_id or --all")

    total = 0
    for gid in game_ids:
        report = retro_sanitize_game(gid, dry_run=args.dry_run)
        for doc, n in report.items():
            marker = "(dry-run) " if args.dry_run else ""
            print(f"  {marker}{gid}/{doc}: {n} picture description(s) cleaned")
            total += n
    if total and not args.dry_run:
        print(f"\n{total} bbox(es) cleaned. Re-index affected games for changes to reach search.")


if __name__ == "__main__":
    _main()
