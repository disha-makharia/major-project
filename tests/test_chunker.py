from backend.rag.chunker import chunk_text


def test_short_text_single_chunk():
    chunks = chunk_text("Hello world.", chunk_size=800, overlap=150)
    assert chunks == ["Hello world."]


def test_long_text_split_into_multiple_chunks():
    paragraph_of = lambda i: f"Sentence number {i}. " * 40
    text = "\n\n".join(paragraph_of(i) for i in range(10))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 50 for c in chunks)  # allow overlap slack


def test_overlap_shares_content_between_chunks():
    paragraphs = [f"Paragraph {i} " + ("word " * 30) for i in range(8)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size=150, overlap=40)
    assert len(chunks) > 1


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []
