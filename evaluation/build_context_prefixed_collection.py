"""Deney: chunk'ları embed ederken "belge başlığı + bölüm başlığı" bağlamı ekleme.

Hipotez: Kısa chunk'lar (bazıları tek cümle) tek başına embed edildiğinde konu bağlamını
kaybediyor, bu da yanlış bölümlerin yüksek skor almasına yol açıyor (bkz. baseline ölçüm,
Recall@1 %56). Bağlam eklenirse (belge başlığı + bölüm başlığı, chunk metninin önüne)
embedding daha isabetli olmalı — ama bu bir HİPOTEZ, aynı 72 soruluk sette yeniden
ölçülecek (evaluation/run_eval.py --collection thy_policies_ctx).

Orijinal (baseline) collection'a dokunulmaz — "thy_policies_ctx" adında AYRI bir
collection oluşturulur, böylece ikisi karşılaştırılabilir (tek değişken: embed metni).

Çalıştırma: PYTHONPATH=. python evaluation/build_context_prefixed_collection.py
"""

from __future__ import annotations

import glob
import re

from app.rag.chunking import Chunk, chunk_markdown_file
from app.rag.qdrant_retriever import QdrantRetriever

TITLE_RE = re.compile(r"^#\s+(.*)")


def load_chunks_with_titles():
    chunks: list[Chunk] = []
    doc_titles: dict[str, str] = {}

    for path in sorted(glob.glob("data/policies/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()

        title_match = TITLE_RE.search(text)
        doc_titles[path] = title_match.group(1).strip() if title_match else path

        chunks.extend(chunk_markdown_file(path, text))

    return chunks, doc_titles


def main() -> None:
    chunks, doc_titles = load_chunks_with_titles()
    print(f"Toplam chunk: {len(chunks)}, belge basligi cikarilan dosya: {len(doc_titles)}")

    def context_prefixed_embed_text(chunk: Chunk) -> str:
        doc_title = doc_titles.get(chunk.source_file, chunk.source_file)
        return f"{doc_title}. {chunk.section_title}. {chunk.text}"

    print("Ornek embed metni (ilk chunk):")
    print(" ", context_prefixed_embed_text(chunks[0])[:200])

    print("\n'thy_policies_ctx' collection'i olusturuluyor ve yukleniyor...")
    QdrantRetriever(
        chunks,
        collection_name="thy_policies_ctx",
        embed_text_fn=context_prefixed_embed_text,
    )
    print("Tamamlandi.")


if __name__ == "__main__":
    main()
