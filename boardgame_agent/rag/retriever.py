"""Qdrant hybrid retrieval for rulebook chunks, filtered by game_id.

Uses Qdrant's native prefetch + RRF fusion to combine:
  - Dense search  (Ollama qwen3-embedding) — semantic similarity
  - Sparse search (SPLADE++) — exact term matching

After RRF fusion, results are optionally re-ranked with a cross-encoder
(Cohere API or local FastEmbed model) for higher precision.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from boardgame_agent.config import (
    COLLECTION_NAME,
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    FASTEMBED_RERANK_MODEL,
    RERANK_PROVIDER,
    RETRIEVAL_TOP_K as _DEFAULT_K,
)
from boardgame_agent.rag.indexer import embed_dense_single, embed_sparse


_cohere_client = None
_fastembed_reranker = None


def _rerank_cohere(query: str, points: list[Any], top_k: int) -> list[Any]:
    """Re-rank using Cohere Rerank API (free tier: 1k calls/month)."""
    global _cohere_client
    if _cohere_client is None:
        import cohere
        _cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

    if not points:
        return points

    documents = [p.payload.get("text", "") for p in points]
    response = _cohere_client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_k,
    )
    return [points[r.index] for r in response.results]


def _rerank_fastembed(query: str, points: list[Any], top_k: int) -> list[Any]:
    """Re-rank using a local FastEmbed cross-encoder model."""
    global _fastembed_reranker
    if _fastembed_reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _fastembed_reranker = TextCrossEncoder(model_name=FASTEMBED_RERANK_MODEL)

    if not points:
        return points

    documents = [p.payload.get("text", "") for p in points]
    results = list(_fastembed_reranker.rerank(query, documents, top_k=top_k))
    return [points[r["index"]] for r in results]


def _rerank(query: str, points: list[Any], top_k: int) -> list[Any]:
    """Re-rank points using the configured provider. Falls back gracefully."""
    if RERANK_PROVIDER == "none" or not points:
        return points[:top_k]
    try:
        if RERANK_PROVIDER == "cohere":
            if not COHERE_API_KEY:
                return points[:top_k]
            return _rerank_cohere(query, points, top_k)
        elif RERANK_PROVIDER == "fastembed":
            return _rerank_fastembed(query, points, top_k)
    except Exception as e:
        print(f"  Re-ranking failed ({RERANK_PROVIDER}): {e} — using RRF order")
    return points[:top_k]


def retrieve_pages(
    client: QdrantClient,
    query: str,
    game_id: str,
    k: int = _DEFAULT_K,
    doc_tag: str | None = None,
) -> list[Any]:
    """Return top-k Qdrant points for *query*, restricted to *game_id*.

    Optionally filter by *doc_tag* (e.g. ``"rulebook"``, ``"faq"``).
    Pass ``None`` to search all documents for the game.

    Pipeline: RRF fusion (dense + sparse, 4×k candidates) → cross-encoder
    re-ranking → final top k.
    """
    conditions = [
        models.FieldCondition(
            key="game_id",
            match=models.MatchValue(value=game_id),
        )
    ]
    if doc_tag is not None:
        conditions.append(
            models.FieldCondition(
                key="doc_tag",
                match=models.MatchValue(value=doc_tag),
            )
        )
    game_filter = models.Filter(must=conditions)

    # Prefetch pool is larger than final k so RRF has enough candidates
    # and the re-ranker has a meaningful pool to work with.
    prefetch_limit = k * 4

    dense_emb = embed_dense_single(query)
    sparse_emb = embed_sparse([query])[0]

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_emb,
                using="dense",
                filter=game_filter,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=sparse_emb,
                using="sparse",
                filter=game_filter,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=prefetch_limit,  # Get full candidate pool for re-ranking
        with_payload=True,
    )

    return _rerank(query, response.points, k)


def format_pages_for_llm(points: list[Any]) -> str:
    """Convert Qdrant points into a structured string the LLM can cite from.

    Points sharing the same (doc_name, page_num) are coalesced into one
    section with a unified bbox list, so the LLM sees a single view of each
    page even when retrieval returns multiple chunks from it.

    Format:
        === DOCUMENT: <doc_name> | PAGE <page_num> ===
        <page text>
        Bboxes (cite by index):
          [0] "..."
          [1] "..."
    """
    if not points:
        return "No relevant pages found in the indexed rulebooks."

    # Group points by (doc_name, page_num), preserving first-seen order so the
    # highest-ranked page stays first in the output.
    groups: dict[tuple[str, Any], list[Any]] = {}
    order: list[tuple[str, Any]] = []
    for point in points:
        p = point.payload
        key = (p.get("doc_name", "unknown"), p.get("page_num", "?"))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(point)

    sections: list[str] = []
    for doc_name, page_num in order:
        chunks = sorted(
            groups[(doc_name, page_num)],
            key=lambda pt: min(pt.payload.get("original_bbox_indices") or [0]),
        )

        text = "\n\n".join(pt.payload.get("text", "") for pt in chunks)

        bbox_by_index: dict[int, dict] = {}
        for pt in chunks:
            pl = pt.payload
            bboxes = pl.get("bboxes", [])
            original_indices = pl.get("original_bbox_indices", list(range(len(bboxes))))
            for i, b in enumerate(bboxes):
                idx = original_indices[i]
                if idx not in bbox_by_index and b.get("text"):
                    bbox_by_index[idx] = b

        bbox_lines = "\n".join(
            f'  [{idx}] "{b.get("text", "")[:200]}"'
            for idx, b in sorted(bbox_by_index.items())
        )

        sections.append(
            f"=== DOCUMENT: {doc_name} | PAGE {page_num} ===\n"
            f"{text}\n\n"
            f"Bboxes (cite by index):\n{bbox_lines}"
        )

    return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(sections)
