"""Deney: chunk boyutunu büyütme (ADR-0004 Yapılacaklar #3, ilk madde).

Hipotez: 600 karakter/100 overlap (~120-150 token) çok küçük kalıyor olabilir; 400-800
token aralığına (~1700-3400 karakter) çıkarmak Recall@1/3'ü artırabilir. Tek değişken:
chunk boyutu (embed stratejisi aynı — context-prefixed, ADR-0002'nin kabul ettiği).

Orijinal "thy_policies" collection'a dokunulmaz — "thy_policies_bigchunk" adında AYRI
bir collection oluşturulur.

Çalıştırma: PYTHONPATH=. python evaluation/build_bigger_chunk_collection.py
"""

from __future__ import annotations

import argparse
import glob
import re

from app.rag.chunking import Chunk, chunk_markdown_file
from app.rag.qdrant_retriever import QdrantRetriever

TITLE_RE = re.compile(r"^#\s+(.*)")


def load_chunks_with_titles(max_chars: int, overlap_chars: int):
    chunks: list[Chunk] = []
    doc_titles: dict[str, str] = {}

    for path in sorted(glob.glob("data/policies/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()

        title_match = TITLE_RE.search(text)
        doc_titles[path] = title_match.group(1).strip() if title_match else path

        chunks.extend(chunk_markdown_file(path, text, max_chars=max_chars, overlap_chars=overlap_chars))

    return chunks, doc_titles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=2500)
    parser.add_argument("--overlap-chars", type=int, default=400)
    parser.add_argument("--collection", default="thy_policies_bigchunk")
    args = parser.parse_args()

    chunks, doc_titles = load_chunks_with_titles(args.max_chars, args.overlap_chars)
    print(f"Toplam chunk: {len(chunks)} (max_chars={args.max_chars}, overlap={args.overlap_chars})")
    lens = [len(c.text) for c in chunks]
    print(f"avg_len={sum(lens)/len(lens):.0f}, min={min(lens)}, max={max(lens)}")

    def context_prefixed_embed_text(chunk: Chunk) -> str:
        doc_title = doc_titles.get(chunk.source_file, chunk.source_file)
        return f"{doc_title}. {chunk.section_title}. {chunk.text}"

    print(f"\n'{args.collection}' collection'i olusturuluyor ve yukleniyor...")
    QdrantRetriever(
        chunks,
        collection_name=args.collection,
        embed_text_fn=context_prefixed_embed_text,
    )
    print("Tamamlandi.")


if __name__ == "__main__":
    main()
