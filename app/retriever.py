"""
retriever.py — Load the persisted Chroma vector store and expose a retriever.

Usage:
    from retriever import get_retriever
    retriever = get_retriever(k=5)
    docs = retriever.invoke("What is quantum entanglement?")
"""

import logging
import sys

from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import settings
from app.utils import setup_logging

logger = setup_logging()


def get_retriever(
    k: Optional[int] = None,
    search_type: Optional[str] = None,
) -> VectorStoreRetriever:
    """
    Load the ChromaDB collection from disk and return a LangChain retriever.

    Args:
        k:           Number of chunks to retrieve (defaults to settings.RETRIEVER_K).
        search_type: "similarity" or "mmr"
                     (defaults to settings.RETRIEVER_SEARCH_TYPE).

    Returns:
        A LangChain ``VectorStoreRetriever`` ready to call `.invoke(query)`.

    Raises:
        SystemExit: If the ChromaDB directory does not exist (run ingest first).
    """
    k = k or settings.RETRIEVER_K
    search_type = search_type or settings.RETRIEVER_SEARCH_TYPE

    if not settings.CHROMA_DIR.exists():
        logger.error(
            "ChromaDB directory not found at './%s'. "
            "Run ingestion first:  python -m app.ingest",
            settings.CHROMA_DIR.name,
        )
        sys.exit(1)

    logger.info("Loading ChromaDB from './%s' …", settings.CHROMA_DIR.name)
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    db = Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.CHROMA_DIR),
    )

    # Sanity check: make sure the collection is non-empty
    count = db._collection.count()
    if count == 0:
        logger.error(
            "ChromaDB collection '%s' is empty. "
            "Run:  python ingest.py",
            settings.COLLECTION_NAME,
        )
        sys.exit(1)

    logger.info("ChromaDB loaded — %d chunk(s) in collection '%s'.", count, settings.COLLECTION_NAME)

    search_kwargs: dict = {"k": k}
    if search_type == "mmr":
        # MMR: diversify results by penalising redundant chunks
        search_kwargs["fetch_k"] = k * 3

    retriever: VectorStoreRetriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    return retriever


def similarity_search(query: str, k: int = 5):
    """
    Convenience function for direct similarity search (not via retriever chain).
    Returns a list of (Document, score) tuples.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    db = Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.CHROMA_DIR),
    )
    return db.similarity_search_with_relevance_scores(query, k=k)
