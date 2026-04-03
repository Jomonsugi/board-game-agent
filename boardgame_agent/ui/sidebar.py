"""Streamlit sidebar: game management, document management, and search domains."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import streamlit as st

from boardgame_agent.config import (
    DATA_DIR,
    DEFAULT_MODEL,
    MODEL_OPTIONS,
    OLLAMA_EMBED_MODEL,
    RETRIEVAL_TOP_K,
    SPARSE_EMBED_MODEL,
    TAVILY_API_KEY,
    VLM_DEFAULT_PRESET,
    VLM_PRESETS,
)
from boardgame_agent.db.games import (
    add_search_domain,
    clear_search_domains,
    create_game,
    delete_document,
    get_all_games,
    get_documents,
    get_search_domains,
    init_db,
    register_document,
    remove_search_domain,
    update_doc_tag,
    update_has_spreads,
    update_vlm_enrichment,
)
from boardgame_agent.rag.extractor import (
    chunk_by_sections,
    get_or_extract,
    load_cached_pages,
    re_enrich_pictures,
)
from boardgame_agent.rag.indexer import build_index, reindex_all, remove_doc_from_index, update_doc_tag_in_index


def _game_id_from_name(name: str) -> str:
    """Convert a game name to a safe identifier."""
    return re.sub(r"[^a-z0-9_]", "_", name.strip().lower()).strip("_")


def render_sidebar() -> tuple[str | None, str | None, str, int, bool]:
    """Render the full sidebar.

    Returns (game_id, game_name, selected_model, top_k, enable_web_search).
    game_id / game_name are None when no game is selected.
    """
    init_db()

    with st.sidebar:
        st.title("Board Game Rules")

        # ── Model settings ────────────────────────────────────────────────────
        with st.expander("Model settings", expanded=False):
            _model_list = list(MODEL_OPTIONS.keys())
            selected_model = st.selectbox(
                "LLM model",
                options=_model_list,
                index=_model_list.index(DEFAULT_MODEL) if DEFAULT_MODEL in _model_list else 0,
                key="selected_model",
            )
            top_k = st.slider(
                "Retrieval pages (top-k)",
                min_value=1,
                max_value=15,
                value=RETRIEVAL_TOP_K,
                step=1,
                key="top_k",
                help="Number of rulebook pages retrieved per query.",
            )
            st.caption(f"**Dense:** `{OLLAMA_EMBED_MODEL}` (Ollama)")
            st.caption(f"**Sparse:** `{SPARSE_EMBED_MODEL}` (SPLADE++)")
            if st.button("Rebuild index (new embed model)", width='stretch'):
                with st.spinner("Rebuilding Qdrant index from cached Docling data…"):
                    reindex_all()
                st.success("Index rebuilt.")

        st.divider()

        # ── Game selector ─────────────────────────────────────────────────────
        games = get_all_games()
        game_names = [g["game_name"] for g in games]
        game_ids = [g["game_id"] for g in games]

        selected_game_name = None
        selected_game_id = None

        if game_names:
            # Apply pending selection from game creation (set before widget renders).
            pending = st.session_state.pop("_pending_game_idx", None)
            if pending is not None:
                st.session_state["selected_game_idx"] = pending

            idx = st.selectbox(
                "Select game",
                options=range(len(game_names)),
                format_func=lambda i: game_names[i],
                key="selected_game_idx",
            )
            selected_game_id = game_ids[idx]
            selected_game_name = game_names[idx]

        # ── Add new game ──────────────────────────────────────────────────────
        with st.expander("Add new game"):
            new_name = st.text_input("Game name", key="new_game_name")
            if st.button("Create game", key="create_game_btn") and new_name.strip():
                gid = _game_id_from_name(new_name)
                create_game(gid, new_name.strip())
                # Queue the new index so the selectbox picks it up on rerun.
                refreshed = get_all_games()
                new_ids = [g["game_id"] for g in refreshed]
                if gid in new_ids:
                    st.session_state["_pending_game_idx"] = new_ids.index(gid)
                st.rerun()

        if selected_game_id is None:
            st.info("Create a game to get started.")
            return None, None, selected_model, top_k, False

        st.divider()

        # ── Documents ─────────────────────────────────────────────────────────
        st.subheader("Documents")
        docs = get_documents(selected_game_id)

        if docs:
            for doc in docs:
                col_name, col_tag, col_spread, col_del = st.columns([4, 3, 2, 1])
                col_name.write(f"📄 {doc['doc_name']}")
                current_tag = doc.get("doc_tag", "rulebook")
                new_tag = col_tag.text_input(
                    "tag",
                    value=current_tag,
                    key=f"tag_{doc['doc_name']}",
                    label_visibility="collapsed",
                    placeholder="rulebook",
                )
                if new_tag != current_tag:
                    update_doc_tag(selected_game_id, doc["doc_name"], new_tag)
                    update_doc_tag_in_index(selected_game_id, doc["doc_name"], new_tag)
                    st.rerun()
                current_spreads = bool(doc.get("has_spreads", 0))
                new_spreads = col_spread.checkbox(
                    "Spreads",
                    value=current_spreads,
                    key=f"spread_{doc['doc_name']}",
                    help="Check if this PDF has two-page spreads (landscape pages with two logical pages side by side).",
                )
                if new_spreads != current_spreads:
                    update_has_spreads(selected_game_id, doc["doc_name"], new_spreads)
                    # Re-extract with the new spread setting and rebuild the index
                    doc_path = Path(doc.get("pdf_path", ""))
                    if doc_path.exists():
                        with st.spinner(f"Re-indexing {doc['doc_name']} with spread {'on' if new_spreads else 'off'}…"):
                            _reindex_doc(selected_game_id, doc["doc_name"], doc_path, current_tag, new_spreads)
                    st.rerun()
                if col_del.button("✕", key=f"del_doc_{doc['doc_name']}", help="Remove"):
                    _remove_document(selected_game_id, doc["doc_name"])
                    st.rerun()

                # VLM picture enrichment (PDF documents only)
                from boardgame_agent.ui.pdf_panel import get_pdf_path
                if get_pdf_path(selected_game_id, doc["doc_name"]):
                    with st.expander("Picture enrichment", expanded=False):
                        current_vlm = doc.get("vlm_model")
                        vlm_labels = list(VLM_PRESETS.keys())
                        default_idx = next(
                            (i for i, k in enumerate(vlm_labels) if VLM_PRESETS[k] == (current_vlm or VLM_DEFAULT_PRESET)),
                            0,
                        )
                        col_vlm, col_btn = st.columns([3, 2])
                        selected_vlm_label = col_vlm.selectbox(
                            "VLM model",
                            options=vlm_labels,
                            index=default_idx,
                            key=f"vlm_{doc['doc_name']}",
                            label_visibility="collapsed",
                        )
                        selected_vlm_preset = VLM_PRESETS[selected_vlm_label]
                        btn_label = "Re-enrich" if current_vlm else "Enrich pictures"
                        if col_btn.button(btn_label, key=f"enrich_{doc['doc_name']}"):
                            with st.spinner(f"Enriching pictures with {selected_vlm_label}…"):
                                count = re_enrich_pictures(
                                    selected_game_id,
                                    doc["doc_name"],
                                    vlm_preset=selected_vlm_preset,
                                    has_spreads=current_spreads,
                                )
                                update_vlm_enrichment(selected_game_id, doc["doc_name"], selected_vlm_preset)
                                # Re-index with enriched text
                                _reindex_after_enrichment(selected_game_id, doc["doc_name"], current_tag)
                            st.success(f"Enriched {count} pictures with {selected_vlm_label}")
                            st.rerun()
                        if current_vlm:
                            enriched_at = doc.get("vlm_enriched_at", "")
                            st.caption(f"Enriched with {current_vlm} ({enriched_at[:10] if enriched_at else ''})")
                    st.rerun()
        else:
            st.caption("No documents indexed yet.")

        # Upload new documents
        uploaded = st.file_uploader(
            "Add document(s)",
            type=["pdf", "md"],
            accept_multiple_files=True,
            key="doc_uploader",
        )
        if uploaded:
            st.caption("Tag each document before indexing:")
            file_tags: dict[str, str] = {}
            file_spreads: dict[str, bool] = {}
            for uf in uploaded:
                col_name, col_tag, col_spread = st.columns([3, 2, 1])
                col_name.write(f"📄 {uf.name}")
                file_tags[uf.name] = col_tag.text_input(
                    "tag",
                    value="rulebook",
                    key=f"upload_tag_{uf.name}",
                    label_visibility="collapsed",
                    placeholder="rulebook",
                )
                file_spreads[uf.name] = col_spread.checkbox(
                    "Spreads",
                    key=f"upload_spread_{uf.name}",
                    help="Check if this PDF has two-page spreads.",
                )
            if st.button(
                f"Index to **{selected_game_name}**",
                key="index_pdfs_btn",
                type="primary",
            ):
                _index_uploaded_docs(selected_game_id, uploaded, file_tags, file_spreads)
                st.rerun()

        # Folder path shortcut (useful for local use)
        folder_path = st.text_input(
            "Or index a folder path", placeholder="/path/to/folder", key="folder_path"
        )
        if folder_path and st.button(
            f"Index folder to **{selected_game_name}**",
            key="index_folder_btn",
        ):
            _index_folder(selected_game_id, Path(folder_path), upload_tag)
            st.rerun()

        st.divider()

        # ── Web search (optional) ─────────────────────────────────────────────
        enable_web_search = False
        if TAVILY_API_KEY:
            enable_web_search = st.checkbox(
                "Enable web search",
                value=True,
                key="enable_web_search",
                help="Requires a Tavily API key. Lets the agent search the web for community rulings.",
            )

            if enable_web_search:
                st.subheader("Web search domains")
                st.caption("Agent searches these sites. Empty = unrestricted.")

                domains = get_search_domains(selected_game_id)
                for domain in domains:
                    col1, col2 = st.columns([4, 1])
                    col1.write(f"🌐 {domain}")
                    if col2.button("✕", key=f"del_dom_{domain}", help="Remove"):
                        remove_search_domain(selected_game_id, domain)
                        st.rerun()

                new_domain = st.text_input("Add domain", placeholder="example.com", key="new_domain")
                col_a, col_b = st.columns(2)
                if col_a.button("Add", key="add_domain_btn") and new_domain.strip():
                    add_search_domain(selected_game_id, new_domain.strip())
                    st.rerun()
                if col_b.button("Clear all", key="clear_domains_btn"):
                    clear_search_domains(selected_game_id)
                    st.rerun()

    return selected_game_id, selected_game_name, selected_model, top_k, enable_web_search


# ── Document management helpers ───────────────────────────────────────────────

_SUPPORTED_EXTENSIONS = {".pdf", ".md"}


def _copy_doc_to_store(game_id: str, src_path: Path, doc_name: str) -> Path:
    """Copy a document into the game's docs directory, preserving extension."""
    ext = src_path.suffix.lower()
    dest_dir = DATA_DIR / "games" / game_id / "docs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{doc_name}{ext}"
    if dest != src_path:
        shutil.copy2(src_path, dest)
    return dest


def _index_single_doc(game_id: str, doc_path: Path, doc_name: str, doc_tag: str = "rulebook", has_spreads: bool = False) -> None:
    """Extract, chunk, embed, and register a single document (PDF or markdown)."""
    stored_path = _copy_doc_to_store(game_id, doc_path, doc_name)
    pages = get_or_extract(stored_path, game_id, doc_name, has_spreads=has_spreads)
    print(f"  {doc_name}: {len(pages)} pages/sections extracted")
    # Inject doc_tag into page dicts so it flows into the Qdrant payload.
    for page in pages:
        page["doc_tag"] = doc_tag
    chunks = chunk_by_sections(pages)
    print(f"  {doc_name}: {len(chunks)} chunks → embedding…")
    build_index(chunks)
    print(f"  {doc_name}: indexing complete")
    cache_path = DATA_DIR / "games" / game_id / "extracted" / f"{doc_name}.json"
    register_document(game_id, doc_name, stored_path, cache_path, doc_tag=doc_tag)


def _index_uploaded_docs(
    game_id: str,
    uploaded_files,
    file_tags: dict[str, str],
    file_spreads: dict[str, bool] | None = None,
) -> None:
    """Index uploaded files with per-file tags.

    *file_tags* maps filename → tag (e.g. ``{"rules.pdf": "rulebook", "faq.md": "faq"}``).
    *file_spreads* maps filename → bool for spread-page PDFs.
    """
    import tempfile, os

    file_spreads = file_spreads or {}
    progress = st.progress(0, text="Indexing…")
    for i, uf in enumerate(uploaded_files):
        doc_name = Path(uf.name).stem
        ext = Path(uf.name).suffix.lower()
        tag = file_tags.get(uf.name, "rulebook")
        spreads = file_spreads.get(uf.name, False)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(uf.read())
            tmp_path = Path(tmp.name)
        try:
            with st.spinner(f"Processing {uf.name}…"):
                _index_single_doc(game_id, tmp_path, doc_name, tag, has_spreads=spreads)
        finally:
            os.unlink(tmp_path)
        progress.progress((i + 1) / len(uploaded_files), text=f"Indexed {uf.name}")
    progress.empty()
    st.success(f"Indexed {len(uploaded_files)} document(s).")


def _index_folder(game_id: str, folder: Path, doc_tag: str = "rulebook") -> None:
    if not folder.is_dir():
        st.error(f"Not a directory: {folder}")
        return
    docs = sorted(f for f in folder.iterdir() if f.suffix.lower() in _SUPPORTED_EXTENSIONS)
    if not docs:
        st.warning("No supported files found (PDF, Markdown).")
        return
    progress = st.progress(0, text="Indexing folder…")
    for i, doc_path in enumerate(docs):
        doc_name = doc_path.stem
        with st.spinner(f"Processing {doc_path.name}…"):
            _index_single_doc(game_id, doc_path, doc_name, doc_tag)
        progress.progress((i + 1) / len(docs), text=f"Indexed {doc_path.name}")
    progress.empty()
    st.success(f"Indexed {len(docs)} document(s) from folder.")


def _reindex_doc(game_id: str, doc_name: str, doc_path: Path, doc_tag: str, has_spreads: bool) -> None:
    """Re-extract and re-index a single document (e.g. after toggling spreads)."""
    # Remove old index entries
    remove_doc_from_index(doc_name, game_id)
    # Delete cached extraction so it re-runs
    cache_path = DATA_DIR / "games" / game_id / "extracted" / f"{doc_name}.json"
    if cache_path.exists():
        cache_path.unlink()
    # Re-extract and re-index
    pages = get_or_extract(doc_path, game_id, doc_name, force=True, has_spreads=has_spreads)
    for page in pages:
        page["doc_tag"] = doc_tag
    chunks = chunk_by_sections(pages)
    build_index(chunks)


def _reindex_after_enrichment(game_id: str, doc_name: str, doc_tag: str) -> None:
    """Re-chunk and re-index after VLM enrichment (no re-extraction needed)."""
    remove_doc_from_index(doc_name, game_id)
    pages = load_cached_pages(game_id, doc_name)
    if pages is None:
        return
    for page in pages:
        page["doc_tag"] = doc_tag
    chunks = chunk_by_sections(pages)
    build_index(chunks)


def _remove_document(game_id: str, doc_name: str) -> None:
    remove_doc_from_index(doc_name, game_id)
    delete_document(game_id, doc_name)
    # Remove cached extraction
    cache_path = DATA_DIR / "games" / game_id / "extracted" / f"{doc_name}.json"
    if cache_path.exists():
        cache_path.unlink()
    # Remove stored document (check both old pdfs/ dir and new docs/ dir)
    for subdir in ("docs", "pdfs"):
        for ext in _SUPPORTED_EXTENSIONS:
            doc_file = DATA_DIR / "games" / game_id / subdir / f"{doc_name}{ext}"
            if doc_file.exists():
                doc_file.unlink()
    st.toast(f"Removed {doc_name}")
