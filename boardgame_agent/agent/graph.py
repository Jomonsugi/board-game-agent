"""LangGraph ReAct agent for the boardgame rules assistant.

Architecture
------------
1. call_agent  — LLM with bound tools (ReAct loop)
2. call_tools  — ToolNode executes requested tool calls
3. finalize    — thin node (no LLM) that parses submit_answer output into state

The graph loops between call_agent and call_tools until the agent calls the
``submit_answer`` tool.  The ``finalize`` node extracts the JSON payload from
that tool's ToolMessage and writes it into ``state["final_answer"]``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_together import ChatTogether
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient

from boardgame_agent.agent.prompts import build_system_prompt
from boardgame_agent.agent.schemas import QAWithCitations
from boardgame_agent.agent.state import AgentState
from boardgame_agent.agent.tools import make_all_tools
from boardgame_agent.config import (
    ANTHROPIC_API_KEY,
    CHECKPOINTS_DB_PATH,
    DEFAULT_MODEL,
    GAMES_DB_PATH,
    MODEL_OPTIONS,
    OPENAI_API_KEY,
    TOGETHER_API_KEY,
)
from boardgame_agent.rag.indexer import get_qdrant_client


_PROVIDER_KEY_MAP = {
    "together": ("TOGETHER_API_KEY", lambda: TOGETHER_API_KEY),
    "anthropic": ("ANTHROPIC_API_KEY", lambda: ANTHROPIC_API_KEY),
    "openai": ("OPENAI_API_KEY", lambda: OPENAI_API_KEY),
}


def _build_llm(model_name: str):
    """Instantiate the correct LangChain chat class based on MODEL_OPTIONS."""
    provider = MODEL_OPTIONS.get(model_name, "together")
    env_name, get_key = _PROVIDER_KEY_MAP[provider]
    key = get_key()
    if not key:
        raise ValueError(
            f"No API key found for {provider}. "
            f"Set {env_name} in your .env file or environment to use {model_name}."
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=key, temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=key, temperature=0)
    else:
        return ChatTogether(model=model_name, together_api_key=key, temperature=0)


def build_agent(
    game_id: str,
    game_name: str,
    model_name: str = DEFAULT_MODEL,
    enable_web_search: bool = True,
) -> tuple[Any, Any, QdrantClient, dict]:
    """Compile the LangGraph agent for *game_id*.

    Returns (compiled_graph, llm, qdrant_client, agent_config).
    *agent_config* is a mutable dict — update ``agent_config["top_k"]``
    before each query so the sidebar slider takes effect without rebuilding.
    """
    from boardgame_agent.config import RETRIEVAL_TOP_K
    from boardgame_agent.db.games import get_documents

    qdrant_client = get_qdrant_client()
    agent_config: dict = {"top_k": RETRIEVAL_TOP_K}
    tools = make_all_tools(
        game_id, game_name, qdrant_client, agent_config, GAMES_DB_PATH,
        enable_web_search=enable_web_search,
    )

    # Build document list for the system prompt.
    docs = get_documents(game_id, GAMES_DB_PATH)
    doc_tuples = [(d["doc_name"], d.get("doc_tag", "rulebook")) for d in docs]

    llm = _build_llm(model_name)
    llm_with_tools = llm.bind_tools(tools)
    system_message = SystemMessage(
        content=build_system_prompt(game_name, documents=doc_tuples, web_search_enabled=enable_web_search)
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def call_agent(state: AgentState) -> dict:
        all_messages = list(state["messages"])

        # Find the last AIMessage so we know which tool outputs have been processed.
        last_ai_idx = max(
            (i for i, m in enumerate(all_messages) if isinstance(m, AIMessage)),
            default=-1,
        )

        # Compress ToolMessages that the LLM has already seen (before last AI turn)
        # to free context space, while preserving tool_call_id pairing.
        compressed: list = []
        for i, m in enumerate(all_messages):
            if isinstance(m, ToolMessage) and i < last_ai_idx:
                compressed.append(
                    ToolMessage(
                        content=f"[retrieved {len(m.content)} chars — already processed]",
                        tool_call_id=m.tool_call_id,
                        name=getattr(m, "name", "tool"),
                    )
                )
            else:
                compressed.append(m)

        response = llm_with_tools.invoke([system_message] + compressed)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def finalize(state: AgentState) -> dict:
        """Extract structured answer from submit_answer tool output (no LLM call).

        Falls back to the agent's last text if submit_answer was not called.
        """
        # Look for the submit_answer ToolMessage
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "submit_answer":
                try:
                    data = json.loads(msg.content)
                    return {"final_answer": data}
                except (json.JSONDecodeError, TypeError):
                    break
            if isinstance(msg, AIMessage):
                break

        # Fallback: agent answered without calling submit_answer
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        return {
            "final_answer": {
                "answer": last_ai.content if last_ai else "No answer produced.",
                "citations": [],
                "web_sources": [],
            }
        }

    # ── Routing ───────────────────────────────────────────────────────────────

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        # Agent responded with text only (no tool calls) — finalize with fallback
        return "finalize"

    def after_tools(state: AgentState) -> str:
        """Route after tool execution: finalize if submit_answer was called."""
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                if getattr(msg, "name", "") == "submit_answer":
                    return "finalize"
            elif isinstance(msg, AIMessage):
                break
        return "agent"

    # ── Graph ─────────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "tools",
        after_tools,
        {"agent": "agent", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    conn = sqlite3.connect(str(CHECKPOINTS_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled, llm, qdrant_client, agent_config


# ── Query helpers ────────────────────────────────────────────────────────────

def _make_input(game_id: str, query: str) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "game_id": game_id,
        "game_name": "",
        "final_answer": None,
    }


def _make_config(thread_id: str | None) -> dict:
    return {
        "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
        "recursion_limit": 25,
    }


def run_query(
    compiled_graph: Any,
    game_id: str,
    query: str,
    thread_id: str | None = None,
) -> QAWithCitations:
    """Invoke the agent (blocking) and return structured QAWithCitations."""
    result = compiled_graph.invoke(
        _make_input(game_id, query),
        config=_make_config(thread_id),
    )
    raw = result.get("final_answer") or {}
    return QAWithCitations(**raw) if raw else QAWithCitations(
        answer="No answer produced.", citations=[]
    )


def run_query_stream(
    compiled_graph: Any,
    game_id: str,
    query: str,
    thread_id: str | None = None,
    on_tool_start: Any = None,
):
    """Stream the agent and call *on_tool_start(tool_name, args)* for each tool.

    Returns the final QAWithCitations when the stream is exhausted.
    """
    final_answer: dict | None = None

    for chunk in compiled_graph.stream(
        _make_input(game_id, query),
        config=_make_config(thread_id),
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            # When the agent node emits tool calls, notify the callback
            if node_name == "agent" and on_tool_start:
                for msg in update.get("messages", []):
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            on_tool_start(tc["name"], tc.get("args", {}))

            # Capture the final answer from the finalize node
            if node_name == "finalize" and update.get("final_answer"):
                final_answer = update["final_answer"]

    if final_answer:
        return QAWithCitations(**final_answer)
    return QAWithCitations(answer="No answer produced.", citations=[])
