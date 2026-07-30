"""TF-IDF baseline ile HF sentence-transformers embedding retriever'ı yan yana karşılaştırma.

Bu bir pytest testi değil, gözlemsel bir demo scripti: amaç, embedding modelinin
Türkçe eş anlamlı/morfolojik varyasyonlarda TF-IDF'e göre gerçekten daha mı iyi
olduğunu VARSAYMADAN, ölçerek görmek (bkz. embeddings.py docstring'i ve
p1_proje_plani.md İlke 2/3: her karar ölçülüp raporlanacak).

Çalıştırma: python evaluation/tfidf_vs_embedding_demo.py
"""

from __future__ import annotations

import glob

from app.rag.chunking import chunk_markdown_file
from app.rag.tfidf_retriever import TfidfRetriever
from app.rag.embeddings import EmbeddingRetriever

QUERIES = [
    ("Kontrol - gecikme (belgedeki kelimeyle birebir)", "Uçuşum 4 saat gecikirse ne hakkım var?"),
    ("Eş anlamlı - 'rötar' (belge 'gecikme/tehir' diyor)", "Uçağım rötar yaparsa tazminat alabilir miyim?"),
    ("Eş anlamlı - 'valiz' + parafraz (belge 'bagaj/oversize' diyor)", "Valizim standarttan büyükse ne kadar ödeme yaparım?"),
    ("Kontrol - check-in (belgedeki kelimeyle birebir)", "Online check-in ne zaman açılıyor?"),
    ("Kapsam dışı - negatif kontrol (corpus'ta yok)", "Evcil hayvanla seyahat edebilir miyim?"),
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
    print(f"Toplam chunk sayisi: {len(chunks)}\n")

    tfidf = TfidfRetriever(chunks)
    print("Ingilizce-agirlikli embedding modeli yukleniyor...")
    embedder_en = EmbeddingRetriever(chunks)
    print("Cok dilli embedding modeli yukleniyor (ilk calistirmada ~470MB indirilir)...")
    embedder_ml = EmbeddingRetriever(
        chunks, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("Modeller hazir.\n")

    methods = [
        ("TF-IDF", tfidf),
        ("Embedding (en)", embedder_en),
        ("Embedding (multi)", embedder_ml),
    ]

    header = f"{'Sorgu tipi':<55} | {'Yontem':<18} | {'Skor':>6} | Bulunan bolum"
    print(header)
    print("-" * len(header))

    for label, query in QUERIES:
        for method_name, retriever in methods:
            result = retriever.search(query, top_k=1)
            title, score = (result[0][0].section_title, result[0][1]) if result else ("-", 0.0)
            print(f"{label:<55} | {method_name:<18} | {score:>6.3f} | {title}")
        print()


if __name__ == "__main__":
    main()
