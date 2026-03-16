"""
config.py — Centralised settings for the Multi-Document RAG system.
All tuneable parameters live here so you never have to hunt through the code.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Base directory (project root) ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """All configuration values, readable from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ── Directories ──────────────────────────────────────────────────────────
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_DIR: Path = BASE_DIR / "chroma_db"

    # ── Chroma collection ────────────────────────────────────────────────────
    COLLECTION_NAME: str = "multi_doc_rag"

    # ── Embedding model (runs locally via sentence-transformers) ─────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── LLM Settings ─────────────────────────────────────────────────────────
    # Which provider to use: "ollama" or "gemini"
    LLM_PROVIDER: str = "ollama"
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_TEMPERATURE: float = 0.0

    # Gemini settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "models/gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.0

    # ── Text splitting ───────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVER_K: int = 5                  # number of chunks to retrieve
    RETRIEVER_SEARCH_TYPE: str = "mmr"   # "similarity" | "mmr"

    RETRIEVER_K: int = 5                  # number of chunks to retrieve
    RETRIEVER_SEARCH_TYPE: str = "mmr"   # "similarity" | "mmr"


# Singleton — import `settings` everywhere
settings = Settings()
