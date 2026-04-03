"""Qdrant indexing for rulebook chunks — hybrid dense + sparse vectors.

Dense vectors  : Ollama qwen3-embedding (4096-d)
Sparse vectors : FastEmbed SPLADE++ (vocabulary-sized, learned term weights)
Fusion         : Qdrant-native RRF at query time (see retriever.py)

build_index  — embed & upsert a list of chunk dicts into the collection.
reindex_all  — rebuild the entire collection from cached Docling JSONs
               (call this after changing embedding models in config.py).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import ollama
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from boardgame_agent.config import (
    COLLECTION_NAME,
    DATA_DIR,
    OLLAMA_EMBED_MODEL,
    OLLAMA_HOST,
    QDRANT_PATH,
    SPARSE_EMBED_MODEL,
)
from boardgame_agent.rag.extractor import chunk_by_sections


# ── Singletons ────────────────────────────────────────────────────────────────

_qdrant_client: QdrantClient | None = None
_ollama_client: ollama.Client | None = None
_sparse_model: SparseTextEmbedding | None = None
_dense_dim: int | None = None


def get_qdrant_client() -> QdrantClient:
    """Return the process-wide shared QdrantClient (created once).

    Cleans up any stale .lock file left by a previous crashed process before
    opening the storage, so the app never requires manual intervention on restart.
    """
    global _qdrant_client
    if _qdrant_client is None:
        lock_file = QDRANT_PATH / ".lock"
        if lock_file.exists():
            lock_file.unlink()
        _qdrant_client = QdrantClient(path=str(QDRANT_PATH))
    return _qdrant_client


def _ensure_ollama_running() -> None:
    """Launch the Ollama macOS app if the server isn't responding."""
    import subprocess
    import time
    import urllib.request

    try:
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=2)
        return  # already running
    except Exception:
        pass

    print("Ollama not running — launching app…")
    subprocess.Popen(["open", "-a", "Ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(30):  # wait up to 15 seconds
        time.sleep(0.5)
        try:
            urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=2)
            print("Ollama is ready.")
            return
        except Exception:
            continue

    raise ConnectionError(
        "Could not start Ollama. Please open the Ollama app manually."
    )


def get_ollama_client() -> ollama.Client:
    """Return the process-wide Ollama client, launching the app if needed."""
    global _ollama_client, _dense_dim
    if _ollama_client is None:
        _ensure_ollama_running()
        _ollama_client = ollama.Client(host=OLLAMA_HOST)
        # Verify connectivity and discover vector dimension.
        test = _ollama_client.embed(model=OLLAMA_EMBED_MODEL, input="test")
        _dense_dim = len(test["embeddings"][0])
    return _ollama_client


def get_dense_dim() -> int:
    """Return the dense embedding dimension (discovered on first Ollama call)."""
    global _dense_dim
    if _dense_dim is None:
        get_ollama_client()
    return _dense_dim


def get_sparse_model() -> SparseTextEmbedding:
    """Return the process-wide FastEmbed sparse model."""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBED_MODEL)
    return _sparse_model


# ── Embedding helpers ─────────────────────────────────────────────────────────

def embed_dense(texts: list[str]) -> list[list[float]]:
    """Embed texts using Ollama. Returns a list of dense float vectors."""
    client = get_ollama_client()
    response = client.embed(model=OLLAMA_EMBED_MODEL, input=texts)
    return response["embeddings"]


def embed_dense_single(text: str) -> list[float]:
    """Embed a single text string using Ollama. Returns one dense vector."""
    return embed_dense([text])[0]


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    """Embed texts using SPLADE++. Returns Qdrant SparseVector objects."""
    sparse_model = get_sparse_model()
    raw = list(sparse_model.embed(texts))
    return [
        models.SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
        for emb in raw
    ]


# ── Collection management ────────────────────────────────────────────────────

def _ensure_collection(client: QdrantClient) -> None:
    """Create the hybrid collection if it doesn't exist yet."""
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=get_dense_dim(),
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )


# ── Index building ───────────────────────────────────────────────────────────

def build_index(
    pages_data: list[dict[str, Any]],
    client: QdrantClient | None = None,
) -> QdrantClient:
    """Embed *pages_data* (dense + sparse) and upsert into Qdrant."""
    if not pages_data:
        return client or get_qdrant_client()

    if client is None:
        client = get_qdrant_client()

    _ensure_collection(client)

    texts = [page["text"] for page in pages_data]
    dense_embeddings = embed_dense(texts)
    sparse_embeddings = embed_sparse(texts)

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            payload=page,
            vector={
                "dense": dense_emb,
                "sparse": sparse_emb,
            },
        )
        for page, dense_emb, sparse_emb in zip(pages_data, dense_embeddings, sparse_embeddings)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return client


# ── Document removal ─────────────────────────────────────────────────────────

def remove_doc_from_index(
    doc_name: str,
    game_id: str,
    client: QdrantClient | None = None,
) -> None:
    """Delete all Qdrant points belonging to *doc_name* in *game_id*."""
    if client is None:
        client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="game_id", match=models.MatchValue(value=game_id)
                    ),
                    models.FieldCondition(
                        key="doc_name", match=models.MatchValue(value=doc_name)
                    ),
                ]
            )
        ),
    )


# ── Tag updates (metadata-only, no re-embedding) ─────────────────────────────

def update_doc_tag_in_index(
    game_id: str,
    doc_name: str,
    doc_tag: str,
    client: QdrantClient | None = None,
) -> None:
    """Update the doc_tag payload on all Qdrant points for a document.

    This is a metadata-only operation — no re-embedding needed.
    """
    if client is None:
        client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return
    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"doc_tag": doc_tag},
        points=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="game_id", match=models.MatchValue(value=game_id)
                    ),
                    models.FieldCondition(
                        key="doc_name", match=models.MatchValue(value=doc_name)
                    ),
                ]
            )
        ),
    )


# ── Full reindex ─────────────────────────────────────────────────────────────

def reindex_all() -> None:
    """Rebuild the entire Qdrant collection from cached Docling JSONs.

    Call this whenever embedding models change in config.py.
    Docling extraction is NOT re-run — only embeddings are rebuilt.
    """
    client = get_qdrant_client()

    # Drop and recreate so stale vectors don't accumulate.
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    games_dir = DATA_DIR / "games"
    if not games_dir.exists():
        print("No games directory found — nothing to reindex.")
        return

    for extracted_dir in sorted(games_dir.glob("*/extracted")):
        game_id = extracted_dir.parent.name
        for json_path in sorted(extracted_dir.glob("*.json")):
            pages = json.loads(json_path.read_text(encoding="utf-8"))
            chunks = chunk_by_sections(pages)
            print(f"  Indexing {game_id}/{json_path.stem} ({len(pages)} pages → {len(chunks)} chunks) …")
            build_index(chunks, client=client)

    print("Reindex complete.")
