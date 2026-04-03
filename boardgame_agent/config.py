"""Central configuration for the boardgame rules agent.

API keys are read from .env or environment variables (e.g. exported in .zshrc).
Everything else is a plain Python constant — edit this file to change defaults.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

# LangSmith: group runs under this project (default is "default" if unset).
os.environ.setdefault("LANGCHAIN_PROJECT", "boardgame_agent")

# ── LLM API keys ──────────────────────────────────────────────────────────────
TOGETHER_API_KEY: str | None = os.getenv("TOGETHER_API_KEY")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# ── Model registry ────────────────────────────────────────────────────────────
# Maps model id → provider. Add Anthropic/OpenAI models here as needed.
# Provider values: "together" | "anthropic" | "openai"
MODEL_OPTIONS: dict[str, str] = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "together",
    "deepseek-ai/DeepSeek-V3.1": "together",
    "claude-sonnet-4-6": "anthropic",
    "gpt-4o": "openai",
}

DEFAULT_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# ── Embeddings ────────────────────────────────────────────────────────────────
# Dense embeddings via Ollama (local). Changing requires "Rebuild index".
OLLAMA_EMBED_MODEL: str = "qwen3-embedding"
OLLAMA_HOST: str = "http://localhost:11434"

# Sparse embeddings via FastEmbed (SPLADE++) for hybrid search.
SPARSE_EMBED_MODEL: str = "prithivida/Splade_PP_en_v1"

# Display name for the sidebar.
EMBED_MODEL_NAME: str = f"{OLLAMA_EMBED_MODEL} + {SPARSE_EMBED_MODEL}"

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Default number of pages retrieved per query. Adjustable in the sidebar.
RETRIEVAL_TOP_K: int = 5

# ── VLM (picture enrichment) ──────────────────────────────────────────────────
# Docling-native VLM presets for describing picture bboxes. All run locally.
# Note: Docling also supports "pixtral" (Pixtral 12B, ~24GB download).
# Omitted from the default list due to size — add it here if the smaller
# models don't produce adequate descriptions for a particular rulebook.
VLM_PRESETS: dict[str, str] = {
    "SmolVLM (256M)": "smolvlm",
    "Granite-Vision (2B)": "granite_vision",
    "Qwen2.5-VL (3B)": "qwen",
}
VLM_DEFAULT_PRESET: str = "qwen"

# Minimum picture area as fraction of page area. Pictures smaller than this
# are skipped during VLM enrichment (e.g., tiny checkbox icons).
# 0.005 = 0.5% of page → keeps ~24x24pt icons, skips ~12x12pt decorations.
VLM_PICTURE_AREA_THRESHOLD: float = 0.005

# ── Web Search (Tavily) ───────────────────────────────────────────────────────
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

# ── Hardware ──────────────────────────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch

    DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
QDRANT_PATH: Path = DATA_DIR / "qdrant"
GAMES_DB_PATH: Path = DATA_DIR / "games.db"
CHECKPOINTS_DB_PATH: Path = DATA_DIR / "agent_checkpoints.db"
COLLECTION_NAME: str = "rulebook_pages"

# Create data directories on import so nothing downstream needs to mkdir.
DATA_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_PATH.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "games").mkdir(exist_ok=True)
