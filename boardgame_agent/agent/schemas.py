"""Shared Pydantic schemas used by the agent and the RAG pipeline."""

from typing import List
from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_name: str = Field(description="Document name exactly as it appears in the DOCUMENT header")
    page_num: int = Field(description="1-indexed page number (or section number for markdown docs)")
    bbox_indices: List[int] = Field(
        default=[],
        description="Indices into that page's bbox array that contain the cited text. Empty list if not applicable.",
    )


class WebSource(BaseModel):
    url: str = Field(description="URL of the web source")
    finding: str = Field(description="One-sentence summary of what was found at this source")


class QAWithCitations(BaseModel):
    answer: str = Field(description="The agent's conversational answer — kept as-is")
    citations: List[Citation] = Field(
        default=[],
        description="Document citations grounding factual claims. Empty for conversational turns.",
    )
    web_sources: List[WebSource] = Field(
        default=[],
        description="Web sources used, each with a URL and a summary of what was found.",
    )
