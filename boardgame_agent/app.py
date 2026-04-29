"""Boardgame Rules Agent — Streamlit app.

Layout
------
  Sidebar  : game selector, document management, web search domains
  Left col : chat interface
  Right col: PDF viewer (scrollable) with highlighted citation overlay
"""

from __future__ import annotations

import logging

# Suppress transformers v5 deprecation spam (one warning per image-processor model).
# These are routed through the logging module, not warnings.warn, so a warnings filter
# does nothing. Filter must be installed before any import that pulls in transformers.
logging.getLogger("transformers").addFilter(
    lambda record: "Accessing `" not in record.getMessage()
)

import uuid

import streamlit as st

from collections import defaultdict

from boardgame_agent.db.games import init_db, save_qa, set_qa_status
from boardgame_agent.agent.graph import build_agent, run_query_stream
from boardgame_agent.agent.schemas import Citation, QAWithCitations
from boardgame_agent.rag.indexer import embed_dense_single
from boardgame_agent.ui.sidebar import render_sidebar
from boardgame_agent.ui.pdf_panel import get_pdf_path, render_highlighted_page, show_pdf_viewer
from boardgame_agent.ui.markdown_panel import get_md_path, render_highlighted_markdown, show_markdown_viewer

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Board Game Rules",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_agent(game_id: str, game_name: str, model_name: str):
    """Build and cache the LangGraph agent.

    Returns (compiled_graph, llm, qdrant_client, agent_config).
    agent_config is a mutable dict — set top_k, enable_web_search, and
    enable_page_vision before each query. These take effect without
    rebuilding the agent or resetting the conversation.
    """
    return build_agent(game_id, game_name, model_name=model_name)


# ── Session state defaults ────────────────────────────────────────────────────

_LAYOUT_PRESETS = {
    "Chat":  [3, 2],
    "Equal": [1, 1],
    "PDF":   [2, 3],
}

def _init_session() -> None:
    defaults = {
        "messages": [],          # list of {"role", "content", "citations"}
        "active_citation": None, # Citation | None
        "active_doc": None,      # doc_name of the PDF currently in the viewer
        "layout": "Equal",       # one of the _LAYOUT_PRESETS keys
        "session_thread_id": str(uuid.uuid4()),  # stable per session, new on restart
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Citation rendering ────────────────────────────────────────────────────────

def _merge_citation_chips(citations: list[dict]) -> list[dict]:
    """Merge citations that share the same (doc_name, page_num).

    Combines bbox_indices arrays and deduplicates. This is a UI-level safety
    net — the submit_answer tool already merges, but legacy data or edge cases
    may still produce duplicates.
    """
    grouped: dict[tuple, list[int]] = defaultdict(list)
    order: list[tuple] = []
    for c in citations:
        key = (c.get("doc_name", ""), c.get("page_num", 0))
        if key not in grouped:
            order.append(key)
        grouped[key].extend(c.get("bbox_indices", []))
    return [
        {"doc_name": doc, "page_num": page, "bbox_indices": sorted(set(grouped[(doc, page)]))}
        for doc, page in order
    ]


def _render_citation_chips(citations: list[dict], game_id: str, msg_idx: int = 0) -> None:
    """Render each citation as a clickable button that updates the PDF panel."""
    if not citations:
        return
    merged = _merge_citation_chips(citations)
    st.markdown("**Citations:**")
    cols = st.columns(min(len(merged), 4))
    for i, c in enumerate(merged):
        doc = c.get("doc_name", "")
        page = c.get("page_num", "?")
        label = f"📄 {doc} · p.{page}"
        with cols[i % 4]:
            if st.button(label, key=f"cite_{msg_idx}_{doc}_{page}_{i}", width='stretch'):
                st.session_state.active_citation = c
                st.session_state.active_doc = doc
                st.rerun()


def _render_accept_buttons(msg: dict) -> None:
    """Render thumbs-up/down feedback for an assistant message.

    Uses Streamlit's native feedback widget for consistent sizing and centering.
    Maps: thumbs-up (1) → accepted, thumbs-down (0) → rejected.
    """
    qa_id = msg.get("qa_id")
    if qa_id is None:
        return

    status = st.session_state.get(f"qa_status_{qa_id}")  # True / False / None
    # st.feedback returns 0 for thumbs-down, 1 for thumbs-up, None if not clicked.
    # Map our stored status to the widget's default index.
    default_index = 1 if status is True else (0 if status is False else None)

    result = st.feedback(
        "thumbs",
        key=f"feedback_{qa_id}",
        disabled=False,
    )

    if result is not None:
        new_status = True if result == 1 else False
        if new_status != status:
            set_qa_status(qa_id, new_status)
            st.session_state[f"qa_status_{qa_id}"] = new_status
            st.rerun()


def _render_web_sources(web_sources: list[dict | str]) -> None:
    """Render web sources with findings and clickable links."""
    if not web_sources:
        return
    st.markdown("**Web sources:**")
    for ws in web_sources:
        if isinstance(ws, dict):
            finding = ws.get("finding", "")
            url = ws.get("url", "")
            st.markdown(f"- {finding} — [{url}]({url})")
        else:
            # Backward compat with old plain-URL format.
            st.markdown(f"- [{ws}]({ws})")


def _render_message(msg: dict, game_id: str, msg_idx: int = 0) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("citations"):
                _render_citation_chips(msg["citations"], game_id, msg_idx=msg_idx)
            if msg.get("web_sources"):
                _render_web_sources(msg["web_sources"])
            _render_accept_buttons(msg)


# ── Document panel ────────────────────────────────────────────────────────────

def _is_pdf_doc(game_id: str, doc_name: str) -> bool:
    return get_pdf_path(game_id, doc_name) is not None


def _is_md_doc(game_id: str, doc_name: str) -> bool:
    return get_md_path(game_id, doc_name) is not None


def _render_doc_panel(game_id: str) -> None:
    citation = st.session_state.active_citation
    active_doc = st.session_state.active_doc

    if citation:
        doc_name = citation.get("doc_name", active_doc)
        page_num = citation.get("page_num", 1)
        bbox_indices = citation.get("bbox_indices", [])

        if _is_pdf_doc(game_id, doc_name):
            st.markdown(f"#### {doc_name} · Page {page_num}")
            img = render_highlighted_page(game_id, doc_name, page_num, bbox_indices)
            if img:
                st.image(img, width='stretch')
            else:
                st.warning("Could not render page — ensure the document is indexed.")

            if st.button("Clear citation", key="clear_citation"):
                st.session_state.active_citation = None
                st.rerun()

            st.divider()
            st.markdown("**Full document:**")
            show_pdf_viewer(game_id, doc_name, scroll_to_page=page_num)

        elif _is_md_doc(game_id, doc_name):
            st.markdown(f"#### {doc_name} · Section {page_num}")
            html = render_highlighted_markdown(game_id, doc_name, page_num, bbox_indices)
            if html:
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.warning("Could not render section — ensure the document is indexed.")

            if st.button("Clear citation", key="clear_citation"):
                st.session_state.active_citation = None
                st.rerun()

            st.divider()
            st.markdown("**Full document:**")
            show_markdown_viewer(game_id, doc_name, scroll_to_section=page_num)

        else:
            st.warning(f"Document file not found for: {doc_name}")

    elif active_doc:
        st.markdown(f"#### {active_doc}")
        if _is_pdf_doc(game_id, active_doc):
            show_pdf_viewer(game_id, active_doc)
        elif _is_md_doc(game_id, active_doc):
            show_markdown_viewer(game_id, active_doc)
    else:
        st.markdown("#### Document Viewer")
        st.info("Click a citation in the chat to view the source with highlights.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    init_db()
    _init_session()

    game_id, game_name, selected_model, top_k, enable_web_search, enable_page_vision = render_sidebar()

    if game_id is None:
        st.markdown("## Welcome to the Board Game Rules Agent")
        st.markdown(
            "Use the sidebar to **create a game** and **add your rulebook PDFs**. "
            "Once indexed, ask any rules question and get cited answers instantly."
        )
        return

    # Clear chat history when the active game changes.
    if st.session_state.get("current_game_id") != game_id:
        st.session_state.messages = []
        st.session_state.active_citation = None
        st.session_state.active_doc = None
        st.session_state.current_game_id = game_id
        st.session_state.current_model = selected_model

    # Reset conversation when model changes (only model — toggles are live).
    model_changed = selected_model != st.session_state.get("current_model")

    if model_changed:
        st.session_state.messages = []
        st.session_state.active_citation = None
        st.session_state.active_doc = None
        st.session_state.session_thread_id = str(uuid.uuid4())
        st.session_state.current_model = selected_model
        st.toast(f"Switched to {selected_model}")
        st.rerun()
    else:
        st.session_state.current_model = selected_model

    # ── Header row: title + layout presets ───────────────────────────────────
    title_col, layout_col = st.columns([3, 1])
    title_col.markdown(f"## {game_name} — Rules Assistant")
    with layout_col:
        chosen = st.radio(
            "Layout",
            options=list(_LAYOUT_PRESETS.keys()),
            index=list(_LAYOUT_PRESETS.keys()).index(st.session_state.layout),
            horizontal=True,
            label_visibility="collapsed",
        )
        if chosen != st.session_state.layout:
            st.session_state.layout = chosen
            st.rerun()

    chat_col, pdf_col = st.columns(_LAYOUT_PRESETS[st.session_state.layout], gap="large")

    # ── Chat column ───────────────────────────────────────────────────────────
    with chat_col:
        # Render conversation history
        for msg_i, msg in enumerate(st.session_state.messages):
            _render_message(msg, game_id, msg_idx=msg_i)

        # Input
        if query := st.chat_input("Ask a rules question…"):
            # Show user message immediately
            st.session_state.messages.append(
                {"role": "user", "content": query, "citations": []}
            )
            with st.chat_message("user"):
                st.markdown(query)

            # Run agent
            compiled, llm, qdrant_client, agent_config = get_agent(
                game_id, game_name, selected_model
            )
            agent_config["top_k"] = top_k
            agent_config["enable_web_search"] = enable_web_search
            agent_config["enable_page_vision"] = enable_page_vision
            # Clear tool call cache for each new query
            agent_config.pop("_tool_cache", None)

            with st.chat_message("assistant"):
                status_container = st.status("Thinking...", expanded=False)

                def _on_tool_start(tool_name: str, args: dict) -> None:
                    if tool_name == "_planner":
                        plan = args.get("plan")
                        if plan:
                            status_container.update(label="Checking conversation context...")
                        # else: no status update needed, move straight to searching
                    elif tool_name == "search_rulebook":
                        source = args.get("source", "all")
                        label = f"Searching documents ({source})..." if source != "all" else "Searching documents..."
                        status_container.update(label=label)
                    elif tool_name == "search_web":
                        status_container.update(label="Searching the web...")
                    elif tool_name == "get_past_answers":
                        status_container.update(label="Checking past answers...")
                    elif tool_name == "view_page":
                        status_container.update(label="Analyzing page visually...")
                    elif tool_name == "submit_answer":
                        status_container.update(label="Preparing answer...")

                qa: QAWithCitations = run_query_stream(
                    compiled, game_id, query,
                    thread_id=st.session_state.session_thread_id,
                    on_tool_start=_on_tool_start,
                )
                status_container.update(label="Done", state="complete")

                st.markdown(qa.answer)

                citations_dicts = [c.model_dump() for c in qa.citations]
                _render_citation_chips(citations_dicts, game_id, msg_idx=len(st.session_state.messages))

                web_source_dicts = [ws.model_dump() if hasattr(ws, 'model_dump') else ws for ws in qa.web_sources]
                if web_source_dicts:
                    _render_web_sources(web_source_dicts)

                # Set the first citation's doc as the active document
                if qa.citations:
                    st.session_state.active_doc = qa.citations[0].doc_name

            # Save to Q&A history (with embedding for future get_past_answers lookups)
            qa_id: int | None = None
            try:
                import numpy as np
                query_emb = np.array(embed_dense_single(query), dtype=np.float32)
                qa_id = save_qa(
                    game_id,
                    query,
                    qa.answer,
                    citations_dicts,
                    embedding=query_emb,
                    model_name=selected_model,
                    top_k=top_k,
                )
                # Seed session status as unreviewed
                st.session_state[f"qa_status_{qa_id}"] = None
            except Exception as e:
                st.warning(f"Could not save Q&A to history: {e}")

            # Persist message with qa_id so accept/reject buttons can reference it
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": qa.answer,
                    "citations": citations_dicts,
                    "web_sources": web_source_dicts,
                    "qa_id": qa_id,
                }
            )

            st.rerun()

    # ── PDF column ────────────────────────────────────────────────────────────
    with pdf_col:
        _render_doc_panel(game_id)


if __name__ == "__main__":
    main()
