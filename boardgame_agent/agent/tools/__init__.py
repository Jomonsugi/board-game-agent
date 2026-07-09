"""Tool registry for the boardgame rules agent.

To add a new tool:
  1. Create agent/tools/your_tool.py with a make_your_tool() factory function.
  2. Import it below and add it to make_all_tools().

That's it — graph.py picks up whatever make_all_tools() returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from qdrant_client import QdrantClient

from boardgame_agent.config import GAMES_DB_PATH
from .rag import make_rag_tool
from .web_search import make_web_search_tool
from .history import make_history_tool
from .submit_answer import make_submit_answer_tool
from .page_vision import make_page_vision_tool
from .lookup_icon import make_lookup_icon_tool


def make_all_tools(
    game_id: str,
    game_name: str,
    qdrant_client: QdrantClient,
    config: dict[str, Any],
    db_path: Path = GAMES_DB_PATH,
) -> list[BaseTool]:
    """Return the complete list of tools available to the agent.

    Tools are instantiated as closures bound to the current game context so
    every tool call is automatically scoped to the right game.

    Web search and page vision are always registered but gated at call time
    via ``config["enable_web_search"]`` and ``config["enable_page_vision"]``.
    This lets the user toggle them mid-conversation without rebuilding the
    agent or losing chat history.
    """
    tools: list[BaseTool] = [
        make_rag_tool(game_id, qdrant_client, config, db_path=db_path),
        make_history_tool(game_id, db_path),
        make_submit_answer_tool(),
        make_page_vision_tool(game_id),
        make_web_search_tool(game_id, db_path, config=config),
        make_lookup_icon_tool(game_id),
    ]
    return tools
