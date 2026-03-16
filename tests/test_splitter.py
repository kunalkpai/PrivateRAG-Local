"""
tests/test_splitter.py — Unit tests for the text-splitting behaviour.

These tests have ZERO external dependencies (no Chroma, no Ollama, no PDFs).
"""

import sys
from pathlib import Path

import pytest

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── Tests ──────────────────────────────────────────────────────────────────

def test_short_text_is_single_chunk(splitter):
    """A text shorter than CHUNK_SIZE should produce exactly one chunk."""
    text = "Hello world. This is a very short document."
    chunks = splitter.split_text(text)
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"


def test_long_text_produces_multiple_chunks(splitter):
    """A text 4× longer than CHUNK_SIZE should produce multiple chunks."""
    word = "word " * 800  # ~4 000 chars >> CHUNK_SIZE=1000
    chunks = splitter.split_text(word)
    assert len(chunks) > 1, "Expected multiple chunks for long text"


def test_no_chunk_exceeds_chunk_size(splitter):
    """Every chunk must respect the CHUNK_SIZE limit."""
    text = "sentence. " * 500  # 5 000 chars
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        assert len(chunk) <= settings.CHUNK_SIZE, (
            f"Chunk {i} has {len(chunk)} chars, exceeds limit {settings.CHUNK_SIZE}"
        )


def test_overlap_content_is_shared(splitter):
    """Consecutive chunks should share some content (overlap > 0)."""
    # Create a text that will be split into at least 2 chunks
    sent = "The quick brown fox jumps over the lazy dog. " * 60  # ~2700 chars
    chunks = splitter.split_text(sent)
    if len(chunks) < 2:
        pytest.skip("Text did not produce enough chunks to test overlap.")

    # The end of chunk[0] and the start of chunk[1] should overlap
    tail = chunks[0][-settings.CHUNK_OVERLAP:]
    head = chunks[1][:settings.CHUNK_OVERLAP]
    # At least SOME characters must be shared
    assert any(c in head for c in tail.split()), (
        "No overlapping content found between consecutive chunks."
    )


def test_empty_text_returns_empty_list(splitter):
    """Empty input should produce an empty list (not crash)."""
    chunks = splitter.split_text("")
    assert chunks == [] or all(c.strip() == "" for c in chunks)


def test_chunk_count_scales_with_length(splitter):
    """More text → more chunks (monotonic relationship)."""
    short = "word " * 100   # ~500 chars
    long  = "word " * 1000  # ~5 000 chars

    short_chunks = splitter.split_text(short)
    long_chunks  = splitter.split_text(long)

    assert len(long_chunks) >= len(short_chunks), (
        "Longer text should produce at least as many chunks as shorter text."
    )
