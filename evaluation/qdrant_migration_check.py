"""Adım 3 doğrulama scripti: tüm chunk'ları Qdrant'a yükler, birkaç kontrol sorusuyla test eder.

Çalıştırma: PYTHONPATH=. python evaluation/qdrant_migration_check.py
Not gerektirir: Qdrant Docker'da çalışıyor olmalı (docker compose -f docker/docker-compose.yml up -d qdrant)
"""

from __future__ import annotations

import glob

from app.rag.chunking import chunk_markdown_file
from app.rag.qdrant_retriever import QdrantRetriever

QUERIES = [
    "Uçuşum 4 saat gecikirse ne hakkım var?",
    "Online check-in ne zaman açılıyor?",
    "Evcil hayvanla seyahat edebilir miyim?",
    "Uçağım rötar yaparsa tazminat alabilir miyim?",
]


def load_chunks():
    chunks = []
    for path in sorted(glob.glob("data/policies/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks.extend(chunk_markdown_file(path, text))
    return chunks


def main() -> None:
    chunks = load_chunks()
    print(f"Toplam chunk sayisi (kaynak): {len(chunks)}")

    print("Qdrant'a yukleniyor (model indirme + embed + upsert, birkac dakika surebilir)...")
    retriever = QdrantRetriever(chunks)

    collection_info = retriever.client.get_collection(retriever.collection_name)
    print(f"Qdrant collection'daki nokta sayisi: {collection_info.points_count}\n")

    for query in QUERIES:
        results = retriever.search(query, top_k=1)
        title, score = (results[0][0].section_title, results[0][1]) if results else ("-", 0.0)
        print(f"Soru: {query}")
        print(f"  -> Bulunan bolum: {title} (skor: {score:.3f})\n")


if __name__ == "__main__":
    main()
