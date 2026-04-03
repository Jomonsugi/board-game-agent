"""SQLite database layer for the boardgame rules agent.

Tables
------
games               — registered games (game_id, game_name, created_at)
documents           — indexed PDFs per game
game_search_domains — per-game allowed web-search domains
qa_history          — past Q&A pairs with question embeddings for semantic lookup
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from boardgame_agent.config import GAMES_DB_PATH


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _connect(db_path: Path = GAMES_DB_PATH):
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def init_db(db_path: Path = GAMES_DB_PATH) -> None:
    """Create all tables if they don't exist."""
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS games (
                game_id    TEXT PRIMARY KEY,
                game_name  TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id            TEXT NOT NULL,
                doc_name           TEXT NOT NULL,
                pdf_path           TEXT NOT NULL,  -- stores path for any doc type (PDF, markdown)
                extracted_json_path TEXT,
                indexed_at         TEXT,
                embed_model        TEXT,
                doc_tag            TEXT NOT NULL DEFAULT 'rulebook',
                has_spreads        INTEGER NOT NULL DEFAULT 0,
                UNIQUE(game_id, doc_name)
            );

            -- Migration: add has_spreads if missing (existing DBs)
            """
        )
        # Check if column exists; add if not.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "has_spreads" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN has_spreads INTEGER NOT NULL DEFAULT 0")
        if "vlm_model" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN vlm_model TEXT")
        if "vlm_enriched_at" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN vlm_enriched_at TEXT")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS game_search_domains (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                domain  TEXT NOT NULL,
                UNIQUE(game_id, domain)
            );

            CREATE TABLE IF NOT EXISTS qa_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id        TEXT NOT NULL,
                question       TEXT NOT NULL,
                answer         TEXT NOT NULL,
                citations_json TEXT NOT NULL,
                embedding      BLOB,
                model_name     TEXT,
                top_k          INTEGER,
                accepted       INTEGER,
                created_at     TEXT NOT NULL
            );
            """
        )
        # Migration: add doc_tag column if missing (added after initial schema).
        try:
            conn.execute("ALTER TABLE documents ADD COLUMN doc_tag TEXT NOT NULL DEFAULT 'rulebook'")
        except sqlite3.OperationalError:
            pass  # column already exists


# ── Games ─────────────────────────────────────────────────────────────────────

def create_game(
    game_id: str,
    game_name: str,
    db_path: Path = GAMES_DB_PATH,
) -> None:
    """Register a new game and seed with the default search domain."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO games (game_id, game_name, created_at) VALUES (?, ?, ?)",
            (game_id, game_name, datetime.utcnow().isoformat()),
        )
        # Default: search boardgamegeek.com only
        conn.execute(
            "INSERT OR IGNORE INTO game_search_domains (game_id, domain) VALUES (?, ?)",
            (game_id, "boardgamegeek.com"),
        )


def get_all_games(db_path: Path = GAMES_DB_PATH) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT game_id, game_name, created_at FROM games ORDER BY game_name"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_game(game_id: str, db_path: Path = GAMES_DB_PATH) -> None:
    """Remove a game and all associated records."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM qa_history WHERE game_id = ?", (game_id,))
        conn.execute("DELETE FROM game_search_domains WHERE game_id = ?", (game_id,))
        conn.execute("DELETE FROM documents WHERE game_id = ?", (game_id,))
        conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))


# ── Documents ─────────────────────────────────────────────────────────────────

def register_document(
    game_id: str,
    doc_name: str,
    pdf_path: Path,
    extracted_json_path: Path | None = None,
    embed_model: str | None = None,
    doc_tag: str = "rulebook",
    db_path: Path = GAMES_DB_PATH,
) -> None:
    from boardgame_agent.config import EMBED_MODEL_NAME

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO documents
                (game_id, doc_name, pdf_path, extracted_json_path, indexed_at, embed_model, doc_tag)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id, doc_name) DO UPDATE SET
                pdf_path           = excluded.pdf_path,
                extracted_json_path = excluded.extracted_json_path,
                indexed_at         = excluded.indexed_at,
                embed_model        = excluded.embed_model,
                doc_tag            = excluded.doc_tag
            """,
            (
                game_id,
                doc_name,
                str(pdf_path),
                str(extracted_json_path) if extracted_json_path else None,
                datetime.utcnow().isoformat(),
                embed_model or EMBED_MODEL_NAME,
                doc_tag,
            ),
        )


def update_has_spreads(
    game_id: str,
    doc_name: str,
    has_spreads: bool,
    db_path: Path = GAMES_DB_PATH,
) -> None:
    """Update the has_spreads flag for a document."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE documents SET has_spreads = ? WHERE game_id = ? AND doc_name = ?",
            (int(has_spreads), game_id, doc_name),
        )


def update_vlm_enrichment(
    game_id: str,
    doc_name: str,
    vlm_model: str,
    db_path: Path = GAMES_DB_PATH,
) -> None:
    """Record that a document was enriched with a VLM model."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE documents SET vlm_model = ?, vlm_enriched_at = ? WHERE game_id = ? AND doc_name = ?",
            (vlm_model, datetime.utcnow().isoformat(), game_id, doc_name),
        )


def update_doc_tag(
    game_id: str,
    doc_name: str,
    doc_tag: str,
    db_path: Path = GAMES_DB_PATH,
) -> None:
    """Update the tag for a document."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE documents SET doc_tag = ? WHERE game_id = ? AND doc_name = ?",
            (doc_tag, game_id, doc_name),
        )


def get_documents(game_id: str, db_path: Path = GAMES_DB_PATH) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE game_id = ? ORDER BY doc_name",
            (game_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(
    game_id: str, doc_name: str, db_path: Path = GAMES_DB_PATH
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM documents WHERE game_id = ? AND doc_name = ?",
            (game_id, doc_name),
        )


# ── Search domains ────────────────────────────────────────────────────────────

def get_search_domains(
    game_id: str, db_path: Path = GAMES_DB_PATH
) -> list[str]:
    """Return allowed domains for *game_id*. Empty list → unrestricted."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT domain FROM game_search_domains WHERE game_id = ? ORDER BY domain",
            (game_id,),
        ).fetchall()
    return [r["domain"] for r in rows]


def add_search_domain(
    game_id: str, domain: str, db_path: Path = GAMES_DB_PATH
) -> None:
    domain = domain.strip().lower()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO game_search_domains (game_id, domain) VALUES (?, ?)",
            (game_id, domain),
        )


def remove_search_domain(
    game_id: str, domain: str, db_path: Path = GAMES_DB_PATH
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM game_search_domains WHERE game_id = ? AND domain = ?",
            (game_id, domain),
        )


def clear_search_domains(game_id: str, db_path: Path = GAMES_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM game_search_domains WHERE game_id = ?", (game_id,)
        )


# ── Q&A History ───────────────────────────────────────────────────────────────

def save_qa(
    game_id: str,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    embedding: np.ndarray | None = None,
    model_name: str | None = None,
    top_k: int | None = None,
    db_path: Path = GAMES_DB_PATH,
) -> int:
    """Persist a Q&A pair and return its row ID."""
    emb_bytes = embedding.astype(np.float32).tobytes() if embedding is not None else None
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO qa_history
                (game_id, question, answer, citations_json, embedding, model_name, top_k, accepted, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                game_id,
                question,
                answer,
                json.dumps(citations),
                emb_bytes,
                model_name,
                top_k,
                datetime.utcnow().isoformat(),
            ),
        )
        return cursor.lastrowid


def set_qa_status(
    qa_id: int,
    accepted: bool | None,
    db_path: Path = GAMES_DB_PATH,
) -> None:
    """Set the accepted status of a Q&A pair.

    accepted=True  → accepted
    accepted=False → rejected
    accepted=None  → reset to unreviewed (NULL)
    """
    value = 1 if accepted is True else (0 if accepted is False else None)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE qa_history SET accepted = ? WHERE id = ?",
            (value, qa_id),
        )


def get_similar_past_answers(
    game_id: str,
    query_embedding: np.ndarray,
    top_k: int = 3,
    db_path: Path = GAMES_DB_PATH,
) -> list[dict[str, Any]]:
    """Return top-k *accepted* Q&A pairs by cosine similarity to *query_embedding*.

    Only rows where accepted = 1 are considered — unreviewed and rejected
    answers are excluded so the agent only builds on verified rulings.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT question, answer, citations_json, embedding FROM qa_history "
            "WHERE game_id = ? AND accepted = 1",
            (game_id,),
        ).fetchall()

    if not rows:
        return []

    results: list[tuple[float, dict]] = []
    q_norm = np.linalg.norm(query_embedding)

    for row in rows:
        if row["embedding"] is None:
            continue
        stored = np.frombuffer(row["embedding"], dtype=np.float32)
        if stored.shape != query_embedding.shape:
            continue
        score = float(np.dot(query_embedding, stored) / (q_norm * np.linalg.norm(stored) + 1e-9))
        results.append(
            (
                score,
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "citations": json.loads(row["citations_json"]),
                },
            )
        )

    results.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in results[:top_k]]


def get_all_qa(game_id: str, db_path: Path = GAMES_DB_PATH) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT question, answer, citations_json, created_at FROM qa_history "
            "WHERE game_id = ? ORDER BY created_at DESC",
            (game_id,),
        ).fetchall()
    return [dict(r) for r in rows]
