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

from boardgame_agent.agent.planner import classify_and_plan
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

# Agent-turn budget. On the Nth turn the agent is forced to answer (submit_answer
# only) instead of searching further; _RECURSION_LIMIT is the hard backstop and
# must leave room for planner + this many agent/tools round-trips (~2 steps each)
# plus the forced turn and finalize.
_SOFT_TURN_CAP = 14
_RECURSION_LIMIT = 40

# Digest budgets for already-seen tool outputs (see call_agent).
_DIGEST_PER_SECTION = 400
_DIGEST_TOTAL = 1600


def _digest_tool_content(content: str) -> str:
    """Shrink an already-seen tool output while keeping it citable.

    search_rulebook output is a series of '=== DOCUMENT: X | PAGE n ===' page
    sections — keep every header plus the head of each section so the model
    still knows which doc/page said what (and doesn't re-retrieve it). Other
    tool outputs keep their head.
    """
    if len(content) <= _DIGEST_TOTAL:
        return content
    marker = "=== DOCUMENT:"
    if marker in content:
        parts = content.split(marker)
        sections = []
        for part in parts[1:]:
            section = marker + part.strip()
            if len(section) > _DIGEST_PER_SECTION:
                section = section[:_DIGEST_PER_SECTION].rstrip() + " …[truncated]"
            sections.append(section)
        digest = "\n\n".join(sections)
    else:
        digest = content[:_DIGEST_TOTAL].rstrip() + " …[truncated]"
    if len(digest) > _DIGEST_TOTAL:
        digest = digest[:_DIGEST_TOTAL].rstrip() + " …[truncated]"
    return "[earlier retrieval, digest] " + digest


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
        # Sonnet 5 / Opus 4.7+ / Fable reject non-default sampling params
        # (temperature=0 -> 400). Omit temperature for those models; older
        # models keep temperature=0 for eval determinism.
        no_sampling = model_name.startswith(
            ("claude-sonnet-5", "claude-opus-4-7", "claude-opus-4-8", "claude-fable")
        )
        kwargs = {} if no_sampling else {"temperature": 0}
        return ChatAnthropic(model=model_name, api_key=key, **kwargs)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=key, temperature=0)
    else:
        # max_tokens matters: Together's default output cap (~2048) gets fully
        # consumed by reasoning models' chain-of-thought (observed: DeepSeek
        # V4-Pro spent 2048/2048 tokens on reasoning at the forced-answer turn
        # and returned empty content). Give synthesis room.
        return ChatTogether(
            model=model_name, together_api_key=key, temperature=0, max_tokens=8192
        )


def build_agent(
    game_id: str,
    game_name: str,
    model_name: str = DEFAULT_MODEL,
) -> tuple[Any, Any, QdrantClient, dict]:
    """Compile the LangGraph agent for *game_id*.

    Returns (compiled_graph, llm, qdrant_client, agent_config).
    *agent_config* is a mutable dict — update ``agent_config["top_k"]``,
    ``agent_config["enable_web_search"]``, and
    ``agent_config["enable_page_vision"]`` before each query so sidebar
    toggles take effect without rebuilding.
    """
    from boardgame_agent.config import RETRIEVAL_TOP_K
    from boardgame_agent.db.games import get_documents

    qdrant_client = get_qdrant_client()
    agent_config: dict = {
        "top_k": RETRIEVAL_TOP_K,
        "enable_web_search": True,
        "enable_page_vision": True,
    }
    all_tools = make_all_tools(
        game_id, game_name, qdrant_client, agent_config, GAMES_DB_PATH,
    )

    llm = _build_llm(model_name)
    # Prompt caching is Anthropic-only. Other providers reject/mishandle the
    # cache_control content-block field, so gate it on the provider.
    supports_prompt_cache = MODEL_OPTIONS.get(model_name, "together") == "anthropic"

    # ── Dynamic tool binding ─────────────────────────────────────────────
    # Tools are bound per invocation based on agent_config toggles.
    # ToolNode keeps all tools registered (for execution), but the LLM
    # only sees currently-enabled tools in its schema — no wasted tokens,
    # no "tool not available" messages the model ignores.
    _TOGGLE_KEYS = {"view_page": "enable_page_vision", "search_web": "enable_web_search"}
    _bind_cache: dict[tuple, object] = {}

    def _get_bound_model():
        """Return the LLM with currently-enabled tools bound.

        Reads agent_config at call time so sidebar toggles take effect
        on the next query without recompiling the graph.
        """
        cache_key = tuple(
            agent_config.get(key, True) for key in _TOGGLE_KEYS.values()
        )
        if cache_key not in _bind_cache:
            active = [
                t for t in all_tools
                if t.name not in _TOGGLE_KEYS
                or agent_config.get(_TOGGLE_KEYS[t.name], True)
            ]
            # strict=True: provider-side schema enforcement of tool args
            # (Anthropic schema adherence / Together-OpenAI strict mode), so
            # required fields like submit_answer.citations can't be omitted.
            _bind_cache[cache_key] = llm.bind_tools(active, strict=True)
        return _bind_cache[cache_key]

    _forced_model_cache: list = []

    def _get_forced_answer_model():
        """LLM bound to only submit_answer, for the soft-stop turn."""
        if not _forced_model_cache:
            submit = next(t for t in all_tools if t.name == "submit_answer")
            _forced_model_cache.append(llm.bind_tools([submit], strict=True))
        return _forced_model_cache[0]

    def _build_system_message(plan: list[str] | None = None) -> SystemMessage:
        """Build the system prompt fresh from the database each call.

        The prompt is byte-stable within a single question (docs are fixed and
        the planner sets `plan` once), so a cache_control breakpoint here lets
        every follow-up turn read the tools+system prefix Anthropic wrote on the
        first turn instead of re-billing it at full price. Render order is
        tools → system → messages, so this one breakpoint caches both the tool
        schemas and the system prompt. Providers that ignore the field (Together,
        OpenAI) just drop it — a list-of-blocks content is still valid there.
        """
        docs = get_documents(game_id, GAMES_DB_PATH)
        doc_tuples = [
            (d["doc_name"], d.get("doc_tag", "rulebook"), d.get("description"))
            for d in docs
        ]
        prompt = build_system_prompt(game_name, documents=doc_tuples, plan=plan)
        if not supports_prompt_cache:
            return SystemMessage(content=prompt)
        return SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def planner(state: AgentState) -> dict:
        """Check if the answer is already in conversation context."""
        return classify_and_plan(state, llm)

    def call_agent(state: AgentState) -> dict:
        all_messages = list(state["messages"])

        # Find the last AIMessage so we know which tool outputs have been processed.
        last_ai_idx = max(
            (i for i, m in enumerate(all_messages) if isinstance(m, AIMessage)),
            default=-1,
        )

        # Compress ToolMessages the LLM has already seen (before last AI turn),
        # preserving tool_call_id pairing. NOT a bare char-count stub: that
        # erased the evidence (doc names, pages, rule text), which forced the
        # model to re-retrieve identical chunks and left the final answer with
        # nothing to cite. The digest keeps each page's identity plus the
        # leading text so earlier findings stay usable and citable.
        compressed: list = []
        for i, m in enumerate(all_messages):
            if isinstance(m, ToolMessage) and i < last_ai_idx:
                compressed.append(
                    ToolMessage(
                        content=_digest_tool_content(str(m.content)),
                        tool_call_id=m.tool_call_id,
                        name=getattr(m, "name", "tool"),
                    )
                )
            else:
                compressed.append(m)

        plan = state.get("plan")
        turns = (state.get("agent_turns") or 0) + 1

        # Soft stop: once the agent has taken many turns without answering, stop
        # letting it search and force a best-effort submit_answer. This converts
        # a hard GraphRecursionError (which yields NO answer and NO citations)
        # into a scoreable answer built from whatever it has already retrieved.
        if turns >= _SOFT_TURN_CAP:
            nudge = HumanMessage(content=(
                "You have reached the search limit and must stop NOW. Do not "
                "call any search or lookup tool again. If the sources you have "
                "already retrieved answer the question, call submit_answer with "
                "that answer and its citations. If they genuinely do NOT, do "
                "not guess and do not present unverified rules as the answer. "
                "Instead call submit_answer with: (1) what you did find and "
                "verify, with citations; (2) what specific information is still "
                "missing; and (3) ONE targeted question for the user whose "
                "answer would let you find it next turn — for example the page "
                "number of the relevant entry, the game edition, or which "
                "expansion is in play. This is an ongoing conversation: the "
                "user can reply, and you will get another chance to answer."
            ))
            response = _get_forced_answer_model().invoke(
                [_build_system_message(plan=plan)] + compressed + [nudge]
            )
        else:
            response = _get_bound_model().invoke(
                [_build_system_message(plan=plan)] + compressed
            )
        return {"messages": [response], "agent_turns": turns}

    tool_node = ToolNode(all_tools)

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
    graph.add_node("planner", planner)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "agent")
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
        "plan": None,
        "agent_turns": 0,
    }


def _make_config(thread_id: str | None) -> dict:
    return {
        "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
        "recursion_limit": _RECURSION_LIMIT,
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
            # When the planner node runs, notify the callback
            if node_name == "planner" and on_tool_start:
                plan = update.get("plan")
                on_tool_start("_planner", {"plan": plan})

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
