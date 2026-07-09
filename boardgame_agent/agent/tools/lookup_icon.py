"""lookup_icon tool — resolved icon meanings from the per-game icon dictionary."""

from __future__ import annotations

from langchain_core.tools import tool


def make_lookup_icon_tool(game_id: str):
    """Return a lookup_icon tool bound to *game_id*.

    Backed by the offline-built icon dictionary (rag/icon_dictionary.py).
    Registered unconditionally; reports cleanly when no dictionary exists so
    the agent falls back to search/vision instead of erroring.
    """

    @tool
    def lookup_icon(query: str) -> str:
        """Look up what a game icon/symbol means in the icon dictionary.

        Use when retrieved text mentions an icon or symbol you need the rule
        meaning of (e.g. an "[Icon: ...]" marker, or a question about what a
        symbol on a card/board does). Returns canonical names, rule meanings,
        and where each icon is defined in the documents.

        Args:
            query: Icon name or meaning keywords (e.g. "order token", "star").
        """
        from boardgame_agent.rag.icon_dictionary import (
            format_icon_text,
            has_dictionary,
            lookup,
        )

        if not has_dictionary(game_id):
            return (
                "No icon dictionary has been built for this game. "
                "Use search_rulebook (and view_page if enabled) instead."
            )
        matches = lookup(game_id, query)
        if not matches:
            return f"No icons matching '{query}' in the dictionary."
        lines = []
        for m in matches[:15]:
            line = format_icon_text(m)
            if m.get("status") == "tentative":
                line += " [tentative — not verified against a definition in the documents]"
            lines.append(line)
        return "\n".join(lines)

    return lookup_icon
