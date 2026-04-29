# Board Game Rules Agent

*A local AI assistant that answers board game rules questions with cited, highlighted references to the official rulebook — built for fast lookups during actual gameplay.*

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)

<!-- TODO: add screenshot here. Suggested shot: chat panel showing an answer with citation chips, alongside the PDF viewer with the cited region highlighted. -->

Ask a question, get an answer with clickable citations that highlight the exact source text in the PDF viewer. The agent cross-references multiple documents when needed and keeps digging until it can ground every claim.

## Quickstart

Prerequisites:
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Ollama](https://ollama.com/download) — for local embeddings: `ollama pull qwen3-embedding`
- A [Together API](https://www.together.ai/) key (free tier works — default LLM provider)

```bash
uv sync
cp .env.example .env
# edit .env and add TOGETHER_API_KEY
uv run boardgame-agent
```

### Optional API keys

| Key | Purpose | Free tier? |
|-----|---------|------------|
| `ANTHROPIC_API_KEY` | Claude models (agent or vision) | No |
| `OPENAI_API_KEY` | GPT-4o models | No |
| `COHERE_API_KEY` | Re-ranking (improves retrieval accuracy) | Yes (1k calls/month) |
| `TAVILY_API_KEY` | Web search fallback | Yes |

## Features

- **Cited, clickable answers.** Every claim links to a paragraph in the source PDF, highlighted in the viewer.
- **Cross-references on its own.** When a supplement points back to a rulebook section, the agent looks it up before answering instead of guessing.
- **Reads icons and diagrams.** A local vision model describes each picture in the rulebook at upload time, so visual elements are searchable alongside the text.
- **Page vision on demand.** For icon-heavy pages where text isn't enough, the agent can visually analyze the rendered page to figure out what to search for next.
- **Web search fallback.** Falls back to trusted domains you configure per game (default: BoardGameGeek) when the indexed documents come up short.
- **Answer history.** Rate answers with thumbs up/down — accepted ones are reused so the agent stays consistent on similar questions.
- **Supports PDF and Markdown.** PDFs get bbox-precise citation highlighting; Markdown gets text-based highlighting.

## Setting up a game

**1. Create a game.** Click *Add new game* in the sidebar.

**2. Upload documents.** Add the rulebook PDF and any supplements (FAQ, player aids, icon references, logbooks). Each document gets:
- a **tag** auto-suggested from the filename — editable any time, no reindexing
- an optional **description** that helps the agent decide when to search this document

**3. Choose processing options.** *Enrich pictures with VLM descriptions* is on by default and makes icons searchable. Uncheck it for text-only rulebooks where pictures don't carry meaning.

**4. Ask questions.** Type a rules question in the chat. Click any citation chip to view the source page with the cited region highlighted.

Per-document settings (description, tag, two-page-spread splitting, re-running picture enrichment) are available in the sidebar under *Options*.

## Built with

LangGraph · Qdrant · Docling · Streamlit · Ollama · Apple MLX

LLM providers are pluggable through LangChain — currently wired to Anthropic, Together, and OpenAI. Map model IDs to providers in `config.py` and switch from the sidebar.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — how the agent reasons, the retrieval pipeline, design decisions.

## License

MIT
