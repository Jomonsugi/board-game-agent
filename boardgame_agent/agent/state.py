"""LangGraph agent state definition."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    game_id: str
    game_name: str
    # Populated after the formatting step; holds QAWithCitations as a plain dict.
    final_answer: Optional[dict[str, Any]]
