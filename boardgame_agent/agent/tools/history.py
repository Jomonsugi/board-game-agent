"""get_past_answers tool — semantic search over per-game Q&A history."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from boardgame_agent.config import GAMES_DB_PATH
from boardgame_agent.rag.indexer import embed_dense_single


def make_history_tool(
    game_id: str,
    db_path: Path = GAMES_DB_PATH,
):
    """Return a get_past_answers tool bound to *game_id*."""

    @tool
    def get_past_answers(query: str) -> str:
        """Check whether a similar rules question has been answered before for
        this game.

        Returns up to 3 previous Q&A pairs ranked by semantic similarity.
        Use this to maintain consistency with prior rulings and to save time
        when the same question recurs.
        """
        import numpy as np
        from boardgame_agent.db.games import get_similar_past_answers

        query_emb = np.array(embed_dense_single(query), dtype=np.float32)
        past = get_similar_past_answers(game_id, query_emb, top_k=3, db_path=db_path)

        if not past:
            return "No previous answers found for this game."

        lines: list[str] = []
        for i, item in enumerate(past, 1):
            lines.append(
                f"[{i}] Q: {item['question']}\n"
                f"    A: {item['answer']}"
            )
        return "\n\n".join(lines)

    return get_past_answers
