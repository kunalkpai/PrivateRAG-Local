"""
tests/test_ingest_integration.py — Integration test for the ingestion pipeline.

Strategy:
  1. Create a tiny synthetic PDF in a temp directory using fpdf2.
  2. Point the ingestion pipeline at that temp dir (monkeypatched settings).
  3. Run ingestion.
  4. Query the resulting Chroma collection and assert results are returned.

Requires: fpdf2  (pip install fpdf2)
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ────────────────────────────────────────────────────────────────

def _create_pdf(path: Path, text: str) -> None:
    """Create a minimal single-page PDF using fpdf2."""
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError:
        pytest.skip("fpdf2 not installed — skipping integration test. Run: pip install fpdf2")

    pdf = FPDF()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.add_page()
    # IMPORTANT: set_font must be called after add_page in fpdf2 >= 2.x
    pdf.set_font("Helvetica", size=11)
    # Write each paragraph; replace empty lines with a small vertical gap
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            pdf.multi_cell(w=0, h=7, txt=stripped, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.ln(4)  # blank line gap
    pdf.output(str(path))


SAMPLE_TEXT = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed "
    "to the natural intelligence displayed by animals including humans. AI research has "
    "been defined as the field of study of intelligent agents, which refers to any "
    "system that perceives its environment and takes actions that maximize its chance of "
    "achieving its goals.\n\n"
    "Machine learning is a subset of AI that gives computers the ability to learn "
    "without being explicitly programmed. Deep learning is a type of machine learning "
    "based on artificial neural networks.\n\n"
    "Large language models (LLMs) are a type of AI that can generate human-like text "
    "and have many applications in natural language processing tasks such as translation, "
    "summarisation, and question answering."
)


# ── Tests ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_env(tmp_path, monkeypatch):
    """
    Provide a temporary data dir + chroma dir and monkeypatch settings.
    Returns (data_dir, chroma_dir).
    """
    data_dir  = tmp_path / "data"
    chroma_dir = tmp_path / "chroma"
    data_dir.mkdir()

    from app import config
    monkeypatch.setattr(config.settings, "DATA_DIR",  data_dir)
    monkeypatch.setattr(config.settings, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(config.settings, "COLLECTION_NAME", "test_collection")

    return data_dir, chroma_dir


def test_ingest_creates_chroma_dir(temp_env):
    """Running ingestion should create the chroma_db directory."""
    data_dir, chroma_dir = temp_env
    _create_pdf(data_dir / "sample.pdf", SAMPLE_TEXT)

    from app.ingest import run_ingestion
    added = run_ingestion()

    assert added > 0, f"Expected chunks to be added, got {added}"
    assert chroma_dir.exists(), "ChromaDB directory was not created"


def test_ingest_returns_correct_chunk_count(temp_env):
    """Chunk count should be positive and consistent with text length."""
    data_dir, _ = temp_env
    _create_pdf(data_dir / "ai_intro.pdf", SAMPLE_TEXT)

    from app.ingest import run_ingestion
    added = run_ingestion()

    # Short text (< 4 000 chars) → at most ~10 chunks
    assert 1 <= added <= 20, f"Unexpected chunk count: {added}"


def test_ingest_idempotent(temp_env):
    """Running ingestion twice on the same file should add 0 new chunks."""
    data_dir, _ = temp_env
    _create_pdf(data_dir / "doc.pdf", SAMPLE_TEXT)

    from app.ingest import run_ingestion
    first_run  = run_ingestion()
    second_run = run_ingestion()

    assert first_run > 0, "First run should add chunks"
    assert second_run == 0, f"Second run should add 0 chunks, got {second_run}"


def test_chroma_retrieval_returns_results(temp_env):
    """After ingestion, a similarity search should return at least one result."""
    data_dir, chroma_dir = temp_env
    _create_pdf(data_dir / "ai.pdf", SAMPLE_TEXT)

    from app.ingest import run_ingestion
    run_ingestion()

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from app import config

    embeddings = HuggingFaceEmbeddings(model_name=config.settings.EMBEDDING_MODEL)
    db = Chroma(
        collection_name="test_collection",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    results = db.similarity_search("What is machine learning?", k=3)
    assert len(results) >= 1, "Expected at least one retrieval result"


def test_ingest_removes_orphaned_documents(temp_env):
    """Removing a file from data_dir should remove its chunks from ChromaDB on next ingest."""
    data_dir, chroma_dir = temp_env
    doc1 = data_dir / "doc1.pdf"
    doc2 = data_dir / "doc2.pdf"
    
    _create_pdf(doc1, "This is the first document.")
    _create_pdf(doc2, "This is the second document.")

    from app.ingest import run_ingestion
    run_ingestion()

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from app import config

    embeddings = HuggingFaceEmbeddings(model_name=config.settings.EMBEDDING_MODEL)
    db = Chroma(
        collection_name="test_collection",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    
    # 1. Both documents should be present
    data = db.get(include=["metadatas"])
    sources = set(m.get("source") for m in data["metadatas"])
    assert "doc1.pdf" in sources
    assert "doc2.pdf" in sources

    # 2. Delete doc1 from the filesystem and re-ingest
    doc1.unlink()
    run_ingestion()

    # 3. doc1 should be gone from the DB
    db = Chroma(
        collection_name="test_collection",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    data = db.get(include=["metadatas"])
    sources = set(m.get("source") for m in data["metadatas"])
    
    assert "doc1.pdf" not in sources, "doc1.pdf should have been purged"
    assert "doc2.pdf" in sources, "doc2.pdf should remain"
