"""
Tests for the RAG pipeline's non-LLM components: chunking, citation
formatting, and retrieval against a real (temporary) Chroma collection.

None of these tests call the Anthropic API -- they cover everything up to
the point where retrieved context would be handed to Claude.

Run with: python -m pytest test_rag.py -v
"""

import shutil
import tempfile

import pytest

from embeddings import TfidfEmbedder
from ingest import chunk_text, ingest
from retrieve import build_context, format_citation, retrieve


# --- chunk_text ---------------------------------------------------------

def test_chunk_text_splits_long_text():
    text = "word " * 500  # 2500 chars, well over CHUNK_SIZE
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert chunks == [text]


def test_chunk_text_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_does_not_split_words():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 5
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    for chunk in chunks:
        assert not chunk.startswith(" ")
        # every chunk should be made of whole words from the source text
        for word in chunk.split():
            assert word in text


def test_chunk_text_rejects_overlap_gte_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, overlap=50)


def test_chunk_text_consecutive_chunks_overlap():
    text = "one two three four five six seven eight nine ten " * 10
    chunks = chunk_text(text, chunk_size=60, overlap=20)
    assert len(chunks) > 1
    # the tail of chunk[0] should share some words with the head of chunk[1]
    tail_words = set(chunks[0].split()[-3:])
    head_words = set(chunks[1].split()[:5])
    assert tail_words & head_words


# --- citation formatting -------------------------------------------------

def test_format_citation():
    chunk = {"text": "irrelevant", "source": "handbook.pdf", "page": 3}
    assert format_citation(chunk) == "[handbook.pdf, p.3]"


def test_build_context_includes_all_chunks_and_citations():
    chunks = [
        {"text": "First chunk text.", "source": "a.pdf", "page": 1},
        {"text": "Second chunk text.", "source": "b.pdf", "page": 5},
    ]
    context = build_context(chunks)
    assert "[a.pdf, p.1]" in context
    assert "[b.pdf, p.5]" in context
    assert "First chunk text." in context
    assert "Second chunk text." in context


def test_build_context_empty_list_returns_empty_string():
    assert build_context([]) == ""


# --- end-to-end retrieval against a real (temporary) Chroma collection ---

@pytest.fixture
def tmp_index():
    """Ingest a small in-memory corpus into a real, temporary Chroma
    collection (via a real PDF-less code path) and clean up afterwards."""
    import os
    from reportlab.pdfgen import canvas

    tmp_dir = tempfile.mkdtemp()
    pdf_path = f"{tmp_dir}/test_doc.pdf"

    c = canvas.Canvas(pdf_path)
    c.drawString(72, 720, "The vacation policy allows 20 days of paid time off per year.")
    c.showPage()
    c.drawString(72, 720, "The parking garage closes at 11pm on weekdays.")
    c.showPage()
    c.save()

    persist_dir = f"{tmp_dir}/chroma_db"
    collection, embedder = ingest([pdf_path], persist_dir=persist_dir, embedder=TfidfEmbedder())

    yield collection, embedder

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_retrieve_returns_relevant_chunk_with_correct_metadata(tmp_index):
    collection, embedder = tmp_index
    results = retrieve("How many vacation days do I get?", collection, embedder, top_k=1)
    assert len(results) == 1
    assert "vacation" in results[0]["text"].lower()
    assert results[0]["page"] == 1


def test_retrieve_respects_top_k(tmp_index):
    collection, embedder = tmp_index
    results = retrieve("policy", collection, embedder, top_k=1)
    assert len(results) == 1
    results = retrieve("policy", collection, embedder, top_k=2)
    assert len(results) == 2
