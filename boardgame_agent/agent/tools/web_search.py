"""search_web tool — Tavily web search with per-game domain restrictions."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from langsmith import traceable

from boardgame_agent.config import TAVILY_API_KEY, GAMES_DB_PATH


def make_web_search_tool(game_id: str, db_path: Path = GAMES_DB_PATH, config: dict | None = None):
    """Return a search_web tool that respects the game's allowed domain list."""

    @tool
    def search_web(query: str) -> str:
        """Search the web for community rulings, FAQs, or clarifications.

        Only use this when the indexed documents do not have the answer, or the
        user explicitly asks you to check the web.

        Results are restricted to trusted domains configured for this game
        (default: boardgamegeek.com). If no domains are configured the search
        is unrestricted.

        Always include the source URL in your answer when citing web results.
        """
        from tavily import TavilyClient
        from boardgame_agent.db.games import get_search_domains

        if not TAVILY_API_KEY:
            return "Web search unavailable — set TAVILY_API_KEY in your .env file."

        # Duplicate call prevention
        if config is not None:
            cache_key = ("search_web", query)
            cache = config.setdefault("_tool_cache", {})
            if cache_key in cache:
                return (
                    "[Cached result — you already ran this exact web search. "
                    "Reformulate your query if you need different results.]\n\n"
                    + cache[cache_key]
                )

        domains = get_search_domains(game_id, db_path)

        @traceable(run_type="tool", name="tavily_search")
        def _tavily_search(query: str, domains: list[str] | None) -> dict:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            kwargs: dict = {
                "query": query,
                "max_results": 5,
                "include_answer": True,
            }
            if domains:
                kwargs["include_domains"] = domains
            return client.search(**kwargs)

        response = _tavily_search(query, domains or None)

        lines: list[str] = []
        if response.get("answer"):
            lines.append(f"Summary: {response['answer']}\n")

        for result in response.get("results", []):
            lines.append(
                f"Source: {result.get('url', 'unknown')}\n"
                f"Title: {result.get('title', '')}\n"
                f"Content: {result.get('content', '')[:600]}\n"
            )

        result_text = "\n---\n".join(lines) if lines else "No results found."

        if config is not None:
            config.setdefault("_tool_cache", {})[("search_web", query)] = result_text

        return result_text

    return search_web
