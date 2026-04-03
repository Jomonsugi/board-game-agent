"""submit_answer tool — the agent's final action to format its answer for the UI.

Instead of a second LLM call to extract citations, the agent calls this tool
directly with structured citation data. The tool validates, merges same-page
citations, and returns a JSON blob that the graph's ``finalize`` node writes
into ``state["final_answer"]``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ── Input schemas (what the model sees in the tool's JSON schema) ────────────


class DocCitation(BaseModel):
    """A citation referencing an indexed document page with optional bounding-box highlights."""

    doc_name: str = Field(description="Document name exactly as shown in the '=== DOCUMENT: ... ===' header")
    page_num: int = Field(description="1-indexed page number from the PAGE field")
    bbox_indices: list[int] = Field(
        default_factory=list,
        description="Bounding-box indices from the 'Bboxes (cite by index)' section of the retrieval output",
    )


class WebSourceCitation(BaseModel):
    """A citation referencing a web search result."""

    url: str = Field(description="Source URL")
    finding: str = Field(description="One-sentence summary of what was found at this source")


class SubmitAnswerInput(BaseModel):
    """Schema for the submit_answer tool call."""

    answer: str = Field(description="Your complete answer to the user's question")
    citations: list[DocCitation] = Field(
        default_factory=list,
        description="Document citations grounding factual claims. Include doc_name, page_num, and bbox_indices.",
    )
    web_sources: Optional[list[WebSourceCitation]] = Field(
        default=None,
        description="Web sources used. Include url and a one-sentence finding for each.",
    )


# ── Merge helper ─────────────────────────────────────────────────────────────


def _merge_citations(citations: list[dict]) -> list[dict]:
    """Merge citations that share the same (doc_name, page_num).

    Combines their ``bbox_indices`` arrays and deduplicates.
    """
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    order: list[tuple[str, int]] = []  # preserve first-seen order

    for c in citations:
        key = (c["doc_name"], c["page_num"])
        if key not in grouped:
            order.append(key)
        grouped[key].extend(c.get("bbox_indices", []))

    merged: list[dict] = []
    for doc_name, page_num in order:
        # Deduplicate and sort bbox indices
        unique_indices = sorted(set(grouped[(doc_name, page_num)]))
        merged.append({
            "doc_name": doc_name,
            "page_num": page_num,
            "bbox_indices": unique_indices,
        })
    return merged


# ── Tool factory ─────────────────────────────────────────────────────────────


def make_submit_answer_tool():
    """Return the submit_answer tool."""

    @tool(args_schema=SubmitAnswerInput)
    def submit_answer(
        answer: str,
        citations: list[dict] | None = None,
        web_sources: list[dict] | None = None,
    ) -> str:
        """Submit your final answer with citations to display in the UI.

        Call this tool ONCE when you have gathered enough information to answer
        the user's question. Pass your answer text, document citations with
        bounding-box indices, and any web sources used.
        """
        raw_citations = [
            c if isinstance(c, dict) else c.model_dump() if hasattr(c, "model_dump") else dict(c)
            for c in (citations or [])
        ]
        raw_web = [
            w if isinstance(w, dict) else w.model_dump() if hasattr(w, "model_dump") else dict(w)
            for w in (web_sources or [])
        ]

        merged = _merge_citations(raw_citations)

        result = {
            "answer": answer,
            "citations": merged,
            "web_sources": raw_web,
        }
        return json.dumps(result)

    return submit_answer
