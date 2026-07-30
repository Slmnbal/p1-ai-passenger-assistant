from app.rag.chunking import chunk_markdown_file

SAMPLE_MD = """# Başlık

## Bölüm Bir
Bu bölümün kısa içeriği.

## Bölüm İki
Birinci paragraf.

İkinci paragraf.
"""


def test_splits_by_headers():
    chunks = chunk_markdown_file("sample.md", SAMPLE_MD)
    titles = [c.section_title for c in chunks]
    assert "Bölüm Bir" in titles
    assert "Bölüm İki" in titles


def test_chunk_ids_are_unique():
    chunks = chunk_markdown_file("sample.md", SAMPLE_MD)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_long_section_is_split_with_overlap():
    long_section = "# Başlık\n\n## Uzun Bölüm\n\n" + "\n\n".join(
        f"Paragraf {i} " + "x" * 50 for i in range(10)
    )
    chunks = chunk_markdown_file("long.md", long_section, max_chars=200, overlap_chars=30)
    assert len(chunks) > 1
    # overlap: art arda gelen chunk'larda ortak metin olmalı
    assert chunks[0].text[-20:] in chunks[1].text or chunks[1].text[:20] in chunks[0].text


def test_empty_document_returns_no_chunks():
    assert chunk_markdown_file("empty.md", "") == []
