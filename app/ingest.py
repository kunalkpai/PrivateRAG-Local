"""
ingest.py — Multi-format document ingestion pipeline.

Supported formats:
  .pdf   PyPDFLoader            (text + page metadata)
  .docx  Docx2txtLoader         (Word documents)
  .txt   TextLoader             (plain text)
  .md    UnstructuredMarkdownLoader  (Markdown files)
  .xlsx  / .xls  — custom pandas loader (each sheet → Document)

Pipeline:
  documents in data/
    → per-format loader  (extract text + metadata)
    → RecursiveCharacterTextSplitter
    → HuggingFaceEmbeddings (all-MiniLM-L6-v2)
    → Chroma (persist to disk)

Run directly:  python ingest.py
Or import:     from ingest import run_ingestion
"""

import logging
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils import setup_logging

logger = setup_logging()


# ── Supported extensions → loader map ─────────────────────────────────────

# Keys are lower-case suffixes (with dot).
# Each value is a callable: path_str -> loader_instance
LOADER_MAP = {
    ".pdf":  lambda p: PyPDFLoader(p),
    ".docx": lambda p: Docx2txtLoader(p),
    ".doc":  lambda p: Docx2txtLoader(p),
    ".txt":  lambda p: TextLoader(p, encoding="utf-8"),
    ".md":   lambda p: UnstructuredMarkdownLoader(p),
    ".xlsx": None,   # handled separately via pandas
    ".xls":  None,   # handled separately via pandas
}

SUPPORTED_EXTENSIONS = set(LOADER_MAP.keys())

# ── Helpers ────────────────────────────────────────────────────────────────

def _load_excel(file_path: Path) -> List[Document]:
    """
    Load an Excel workbook (.xlsx / .xls).
    Each sheet is converted to a CSV-style text block and returned as
    one Document, with metadata: source, sheet_name.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        logger.warning(
            "pandas / openpyxl not installed — skipping '%s'. "
            "Run:  pip install pandas openpyxl",
            file_path.name,
        )
        return []

    try:
        xl = pd.ExcelFile(file_path)
    except Exception as exc:
        logger.error("Failed to open Excel file '%s': %s", file_path.name, exc)
        return []

    docs: List[Document] = []
    for sheet_name in xl.sheet_names:
        try:
            df = xl.parse(sheet_name)
            # Drop entirely empty rows / columns
            df.dropna(how="all", inplace=True)
            df.dropna(axis=1, how="all", inplace=True)
            if df.empty:
                continue
            text = df.to_csv(index=False)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "sheet_name": str(sheet_name),
                        "page": 0,
                    },
                )
            )
        except Exception as exc:
            logger.warning("Could not read sheet '%s' in '%s': %s", sheet_name, file_path.name, exc)

    logger.info("  → %d sheet(s) extracted from '%s'", len(docs), file_path.name)
    return docs


def _load_documents(data_dir: Path) -> List[Document]:
    """
    Scan *data_dir* for all supported file types and load them.

    Supported: .pdf  .docx  .doc  .txt  .md  .xlsx  .xls
    """
    all_docs: List[Document] = []
    found_files = [
        p for p in sorted(data_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not found_files:
        logger.warning(
            "No supported documents found in '%s'. "
            "Supported: %s",
            data_dir,
            ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )
        return []

    for file_path in found_files:
        suffix = file_path.suffix.lower()
        logger.info("Loading  %-45s [%s] ...", file_path.name, suffix)

        # Excel uses the custom pandas loader
        if suffix in (".xlsx", ".xls"):
            docs = _load_excel(file_path)
        else:
            loader_factory = LOADER_MAP.get(suffix)
            if loader_factory is None:
                logger.warning("No loader for '%s' — skipping.", file_path.name)
                continue
            try:
                loader = loader_factory(str(file_path))
                docs = loader.load()
            except Exception as exc:
                logger.error("Failed to load '%s': %s", file_path.name, exc)
                continue

        # Normalise source metadata to filename only
        for doc in docs:
            doc.metadata["source"] = file_path.name
            # Ensure a 'page' key always exists for consistent citation display
            doc.metadata.setdefault("page", 0)

        all_docs.extend(docs)
        logger.info("  → %d document section(s) extracted", len(docs))

    logger.info("Total sections loaded: %d (from %d file(s))", len(all_docs), len(found_files))
    return all_docs


def _split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        # Prefer splitting on paragraph / sentence boundaries first
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        "Split %d page(s) into %d chunk(s)  (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
    )
    return chunks


def _get_existing_sources(chroma_dir: Path, embeddings) -> set:
    """Return a set of source filenames already stored in ChromaDB."""
    if not chroma_dir.exists():
        return set()

    try:
        db = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )
        data = db.get(include=["metadatas"])
        sources = {m.get("source", "") for m in data["metadatas"]}
        return sources
    except Exception:
        return set()


def _purge_orphaned_sources(chroma_dir: Path, embeddings, orphaned_sources: set) -> None:
    """Remove chunks from ChromaDB whose source files no longer exist."""
    if not orphaned_sources or not chroma_dir.exists():
        return

    try:
        db = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )
        for source in orphaned_sources:
            logger.info("Purging removed document from DB: '%s'", source)
            # Use raw chromadb collection API for precise deletion by metadata
            db._collection.delete(where={"source": source})
    except Exception as exc:
        logger.error("Failed to purge orphaned sources: %s", exc)


# ── Public API ─────────────────────────────────────────────────────────────

def run_ingestion() -> int:
    """
    Full ingestion pipeline.

    Returns:
        int: Number of NEW chunks added to the vector store.
    """
    logger.info("=== Starting ingestion pipeline ===")
    logger.info("Data dir  : ./%s", settings.DATA_DIR.name)
    logger.info("Chroma dir: ./%s", settings.CHROMA_DIR.name)
    logger.info("Embedding : %s", settings.EMBEDDING_MODEL)

    # 1. Embedding model (downloads on first run, then cached)
    logger.info("Loading embedding model '%s' …", settings.EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # 2. Find which PDFs are already ingested (idempotency)
    existing_sources = _get_existing_sources(settings.CHROMA_DIR, embeddings)
    
    # 3. Detect and purge orphaned sources (files deleted from data/)
    if settings.DATA_DIR.exists():
        current_files = {
            p.name for p in settings.DATA_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        }
        orphaned_sources = existing_sources - current_files
        if orphaned_sources:
            _purge_orphaned_sources(settings.CHROMA_DIR, embeddings, orphaned_sources)
            existing_sources = existing_sources - orphaned_sources

    if existing_sources:
        logger.info("Already in DB: %s", ", ".join(sorted(existing_sources)))

    # 4. Load all supported documents
    all_docs = _load_documents(settings.DATA_DIR)
    if not all_docs:
        return 0

    # 4. Filter out already-ingested documents
    new_docs = [d for d in all_docs if d.metadata.get("source", "") not in existing_sources]
    if not new_docs:
        logger.info("All documents are already indexed. Nothing to do.")
        return 0

    new_sources = {d.metadata["source"] for d in new_docs}
    logger.info("New files to ingest: %s", ", ".join(sorted(new_sources)))

    # 5. Split into chunks
    chunks = _split_documents(new_docs)
    if not chunks:
        logger.info("No valid text chunks found in new documents. Skipping insertion.")
        return 0

    # 6. Embed + persist in Chroma
    settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Embedding %d chunk(s) and writing to ChromaDB …", len(chunks))

    if existing_sources:
        # Add to existing collection
        db = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(settings.CHROMA_DIR),
        )
        db.add_documents(chunks)
    else:
        # Create new collection
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=settings.COLLECTION_NAME,
            persist_directory=str(settings.CHROMA_DIR),
        )

    logger.info("✅ Ingestion complete — %d chunk(s) added.", len(chunks))
    return len(chunks)


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    added = run_ingestion()
    print(f"\nDone. {added} chunk(s) stored in ChromaDB.")
