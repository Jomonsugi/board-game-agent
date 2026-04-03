# Board Game Rules Agent

A local AI assistant that answers board game rules questions with **cited, highlighted** references to the official rulebook — built for fast lookups during actual gameplay.

## Quick start

Install prerequisites:
- [uv](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Ollama](https://ollama.com/download): download the macOS app, then `ollama pull qwen3-embedding`
- A [Together API](https://www.together.ai/) key (free tier works — this is the default LLM provider)

Then:

```bash
cd boardgame_agent
uv sync
cp .env.example .env
# Edit .env and add your TOGETHER_API_KEY
boardgame-agent
```

Create a game in the sidebar, upload a rulebook PDF, and ask a question. That's it.

---

## Using the app

**Create a game and add documents.** Click **Add new game** — the new game is auto-selected. Upload PDFs or markdown files, or point to a folder. Each document gets a **tag** (default "rulebook") that you can edit anytime in the sidebar. Docling parses PDFs once (can take a few minutes for large rulebooks). The first query also downloads the SPLADE++ sparse model (~530 MB, one-time).

**Document tags.** Tags tell the agent what kind of document it's searching. The default is "rulebook" — use any label you want for additional documents (faq, errata, supplement, etc.). The agent sees all tagged documents and searches rulebook-tagged ones first, then consults others when needed. Tags are editable inline — changes apply instantly, no reindexing.

**Ask questions.** Type a rules question in the chat. The agent searches indexed documents, retrieves relevant pages, and returns a cited answer. Click any **citation chip** to view the source — PDFs show highlighted page images, markdown files show highlighted text.

**Rate answers.** Each response has ✅ and ❌ buttons. Accepted answers feed into the `get_past_answers` tool so the agent stays consistent with prior verified rulings. Click again to undo.

**Top-k slider.** Adjusts how many pages are retrieved per query. Takes effect immediately — no session reset.

**Web search (optional).** Requires a `TAVILY_API_KEY` in `.env`. When set, a checkbox appears in the sidebar to enable/disable web search. Add trusted domains (e.g., `boardgamegeek.com`) to restrict where the agent searches.

**Switching LLM models.** Use the dropdown in the sidebar. Changing the model resets the current conversation (you'll be warned first).

**Rebuild index.** After changing the embedding model in `config.py`, click **Rebuild index** in the sidebar. This re-embeds all cached documents — extraction does not re-run.

## LLM providers

The default models use Together API, but you can use Anthropic, OpenAI, or any combination. Models and their providers are configured in `config.py` under `MODEL_OPTIONS` — map each model ID to `"together"`, `"anthropic"`, or `"openai"`. Only add API keys for the providers you use. If a key is missing when you select a model, you'll get a clear error telling you which key to set.

## Embeddings

Dense vectors via Ollama (default `qwen3-embedding`, 4096-d). Sparse vectors via FastEmbed SPLADE++. Results are fused with Qdrant-native RRF hybrid search. Any Ollama embedding model can be used — change `OLLAMA_EMBED_MODEL` in `config.py` and click **Rebuild index**.

Ollama launches automatically if the app is installed but not running.

## Supported document formats

- **PDF** — parsed by Docling with full bounding-box citations and highlighted page rendering
- **Markdown** (.md) — parsed by heading structure with text-based citation highlighting

Both formats are indexed identically (same hybrid dense + sparse vectors) and are searchable through the same tool. Adding new formats in the future requires only a new extractor — no reindexing of existing documents.

## Project structure

```
boardgame_agent/
├── app.py              # Streamlit entry point
├── config.py           # All tunable settings
├── agent/
│   ├── graph.py        # LangGraph ReAct agent
│   ├── prompts.py      # Dynamic system prompt with document awareness
│   ├── schemas.py      # QAWithCitations, Citation
│   ├── state.py        # AgentState
│   └── tools/
│       ├── __init__.py # Tool registry
│       ├── rag.py      # search_rulebook (hybrid, filterable by tag)
│       ├── web_search.py # search_web (Tavily, optional)
│       └── history.py  # get_past_answers
├── rag/
│   ├── extractor.py    # Format dispatch + Docling PDF extraction
│   ├── markdown_extractor.py  # Markdown parsing into page dicts
│   ├── indexer.py      # Qdrant hybrid indexing (Ollama + SPLADE++)
│   └── retriever.py    # Hybrid retrieval with RRF fusion + tag filtering
├── db/
│   └── games.py        # SQLite: games, documents, domains, Q&A history
├── ui/
│   ├── pdf_panel.py    # PyMuPDF highlights + PDF viewer
│   ├── markdown_panel.py  # Markdown citation highlights + viewer
│   └── sidebar.py      # Game & document management UI
└── data/               # Runtime data (gitignored)
    ├── qdrant/
    ├── games.db
    └── games/{game_id}/
        ├── docs/       # Stored documents (PDF, markdown)
        └── extracted/  # Cached extraction JSON
```
