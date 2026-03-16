"""
utils.py — Shared helpers: logging, display formatting.
"""

import logging
import sys
from typing import List

from langchain_core.documents import Document


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger and return a named logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Suppress verbose INFO logs from external networking/AI libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    # Suppress Pydantic V1 deprecation warnings from LangChain in Python 3.12+
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    
    return logging.getLogger("rag")


def format_sources(docs: List[Document]) -> str:
    """
    Pretty-print retrieved source chunks.

    Returns a multi-line string like:
        📄 sample.pdf  (page 3)
           "The quick brown fox…"
    """
    if not docs:
        return "  (no sources retrieved)"

    lines: List[str] = []
    seen: set = set()

    for doc in docs:
        meta = doc.metadata
        source = meta.get("source", "unknown")
        
        # Convert 0-indexed page to 1-indexed for user-friendly display
        raw_page = meta.get("page", 0)
        try:
            page = int(raw_page) + 1
        except (ValueError, TypeError):
            page = raw_page
            
        key = (source, page)

        if key not in seen:
            seen.add(key)
            snippet = doc.page_content[:120].replace("\n", " ").strip()
            lines.append(f"  📄 {source}  (page {page})")
            lines.append(f'     "{snippet}…"')

    return "\n".join(lines)


def print_banner() -> None:
    """Print a startup banner."""
    print(
        "\n"
        "╔══════════════════════════════════════════════╗\n"
        "║   🔍  Multi-Document RAG  ·  Powered by       ║\n"
        "║       LangChain + Ollama + ChromaDB           ║\n"
        "╚══════════════════════════════════════════════╝\n"
    )
