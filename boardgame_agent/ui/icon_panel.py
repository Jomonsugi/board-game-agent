"""Streamlit sidebar section: per-game icon dictionary (build, review, apply).

Human-in-the-loop review lives here: the auto-built dictionary is a small
table (~15–30 icons per game), so a couple of minutes of review buys a lot of
reliability. Reviewed entries are never overwritten by later resolve runs.
"""

from __future__ import annotations

import streamlit as st

from boardgame_agent.config import DATA_DIR, ICON_RESOLVE_MODEL, ICON_VLM_OPTIONS
from boardgame_agent.rag.icon_dictionary import (
    APPLIABLE_STATUSES,
    apply_to_cache,
    build_icon_dictionary,
    connect,
    get_stats,
    has_dictionary,
)

_STATUS_BADGES = {
    "resolved": "🟢 resolved",
    "reviewed": "✅ reviewed",
    "tentative": "🟡 tentative",
    "unresolved": "🔴 unresolved",
    "new": "⚪ new",
}


def render_icon_dictionary_section(game_id: str, game_name: str) -> None:
    """Render the icon-dictionary expander for the selected game."""
    with st.expander("Icon dictionary", expanded=False):
        stats = get_stats(game_id)

        if stats is None:
            st.caption(
                "Optional, for games whose questions hinge on iconography. "
                "Builds an offline dictionary of icon meanings that gets "
                "injected into search."
            )
        else:
            by_status = stats["by_status"]
            parts = [
                f"{n} {status}" for status, n in sorted(by_status.items()) if n
            ]
            st.caption(
                f"**{stats['icons']} icons** ({', '.join(parts)}) · "
                f"{stats['instances']} placements"
            )

        # ── Build / update ────────────────────────────────────────────────
        model_list = list(ICON_VLM_OPTIONS.keys())
        default_idx = (
            model_list.index(ICON_RESOLVE_MODEL)
            if ICON_RESOLVE_MODEL in model_list else 0
        )
        model = st.selectbox(
            "Resolve model (vision)",
            options=model_list,
            index=default_idx,
            key=f"icon_model_{game_id}",
            help="Frontier VLM used to resolve icon meanings. Reviewed entries "
                 "are never overwritten, so trying a better model later is safe.",
        )

        btn_label = "Update dictionary" if stats else "Build dictionary"
        if st.button(btn_label, key=f"icon_build_{game_id}"):
            with st.status("Building icon dictionary…", expanded=True) as status:
                report = build_icon_dictionary(
                    game_id, model=model, progress=st.write
                )
                status.update(state="complete", expanded=False)
            r = report["resolve"]
            st.success(
                f"{report['dedupe']['icons']} icons — "
                f"{r['resolved']} resolved, {r['tentative']} tentative, "
                f"{r['unresolved']} unresolved, "
                f"{report['consolidate']['merged']} duplicate(s) merged. "
                f"Review below, then apply."
            )
            st.rerun()

        if stats is None:
            return

        # ── Review ────────────────────────────────────────────────────────
        conn = connect(game_id)
        try:
            icons = conn.execute(
                "SELECT * FROM icons ORDER BY "
                "CASE status WHEN 'unresolved' THEN 0 WHEN 'tentative' THEN 1 "
                "WHEN 'resolved' THEN 2 WHEN 'new' THEN 3 ELSE 4 END, "
                "n_instances DESC"
            ).fetchall()

            for icon in icons:
                iid = icon["icon_id"]
                col_img, col_fields = st.columns([1, 4])
                crop = DATA_DIR / icon["crop_path"]
                if crop.exists():
                    col_img.image(str(crop), width=48)
                col_img.caption(f"×{icon['n_instances']}")

                name = col_fields.text_input(
                    "name", value=icon["name"] or "",
                    key=f"icon_name_{iid}", label_visibility="collapsed",
                    placeholder="canonical name",
                )
                meaning = col_fields.text_area(
                    "meaning", value=icon["meaning"] or "",
                    key=f"icon_meaning_{iid}", label_visibility="collapsed",
                    placeholder="the rule this icon encodes", height=68,
                )
                badge = _STATUS_BADGES.get(icon["status"], icon["status"])
                if icon["def_doc"]:
                    badge += f" · defined in {icon['def_doc']} p.{icon['def_page']}"
                col_a, col_b = col_fields.columns([3, 1])
                col_a.caption(badge)

                edited = (name != (icon["name"] or "")) or (meaning != (icon["meaning"] or ""))
                if col_b.button(
                    "Save" if edited else "Approve",
                    key=f"icon_approve_{iid}",
                    disabled=(not meaning.strip()),
                ):
                    conn.execute(
                        "UPDATE icons SET name = ?, meaning = ?, status = 'reviewed' "
                        "WHERE icon_id = ?",
                        (name.strip(), meaning.strip(), iid),
                    )
                    conn.commit()
                    st.rerun()
        finally:
            conn.close()

        # ── Apply ─────────────────────────────────────────────────────────
        st.divider()
        n_appliable = sum(
            stats["by_status"].get(s, 0) for s in APPLIABLE_STATUSES
        )
        if st.button(
            f"Apply {n_appliable} meaning(s) to index",
            key=f"icon_apply_{game_id}",
            type="primary",
            disabled=(n_appliable == 0),
            help="Injects icon meanings into the extraction cache and re-indexes "
                 "this game's documents so icon questions become plain text search.",
        ):
            with st.spinner("Injecting meanings and re-indexing…"):
                report = apply_to_cache(game_id)
                _reindex_game_docs(game_id, [d for d, n in report.items()])
            st.success(
                "Applied: " + ", ".join(f"{d} ({n})" for d, n in report.items())
            )


def _reindex_game_docs(game_id: str, doc_names: list[str]) -> None:
    """Re-chunk and re-index documents from their (now icon-enriched) caches."""
    from boardgame_agent.db.games import get_documents
    from boardgame_agent.rag.extractor import chunk_by_sections, load_cached_pages
    from boardgame_agent.rag.indexer import build_index, remove_doc_from_index

    tags = {d["doc_name"]: d.get("doc_tag", "rulebook") for d in get_documents(game_id)}
    for doc_name in doc_names:
        pages = load_cached_pages(game_id, doc_name)
        if pages is None:
            continue
        remove_doc_from_index(doc_name, game_id)
        for page in pages:
            page["doc_tag"] = tags.get(doc_name, "rulebook")
        build_index(chunk_by_sections(pages))
