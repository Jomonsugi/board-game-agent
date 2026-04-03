"""search_rulebook tool — hybrid RAG retrieval from indexed documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from qdrant_client import QdrantClient

from boardgame_agent.config import GAMES_DB_PATH
from boardgame_agent.rag.retriever import retrieve_pages, format_pages_for_llm


def make_rag_tool(
    game_id: str,
    qdrant_client: QdrantClient,
    config: dict[str, Any],
    db_path: Path = GAMES_DB_PATH,
):
    """Return a search_rulebook tool bound to *game_id*.

    *config* is a mutable dict — ``config["top_k"]`` is read at call time
    so the sidebar slider takes effect without rebuilding the agent.
    """

    @tool
    def search_rulebook(query: str, source: str = "all") -> str:
        """Search the indexed documents for rules relevant to the query.

        Always call this tool first for any rules question. Returns page text
        and numbered bounding-box references you must use in citations.

        Args:
            query: The search query.
            source: Filter by document tag. Use a specific tag like 'rulebook'
                    or 'faq' to search only those documents, or 'all' to search
                    everything.
        """
        cache_key = ("search_rulebook", query, source)
        cache = config.setdefault("_tool_cache", {})
        if cache_key in cache:
            return (
                "[Cached result — you already ran this exact search. "
                "Reformulate your query or try a different source.]\n\n"
                + cache[cache_key]
            )

        doc_tag = None if source == "all" else source

        # Validate the tag before searching.
        if doc_tag is not None:
            from boardgame_agent.db.games import get_documents
            docs = get_documents(game_id, db_path)
            known_tags = sorted(set(d.get("doc_tag", "rulebook") for d in docs))
            if doc_tag not in known_tags:
                result = (
                    f"Tag '{doc_tag}' does not exist for this game. "
                    f"Available tags: {known_tags}. "
                    f"Use one of these, or source='all' to search everything."
                )
                cache[cache_key] = result
                return result

        points = retrieve_pages(qdrant_client, query, game_id, k=config["top_k"], doc_tag=doc_tag)
        result = format_pages_for_llm(points)
        cache[cache_key] = result
        return result

    return search_rulebook
