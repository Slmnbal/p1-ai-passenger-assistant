"""Hipotez testi: embedding'in top-1 sonucu ile sorgu arasindaki ortak anlamli kelime
sayisi, kapsam-disi (negatif) sorulari kontrol/pozitif sorulardan ayirt edebiliyor mu?

Bu bir "duzeltme" degil, saf bir olcum/analiz scripti — once hipotezi dogrula, sonra
(isterse) bir guardrail kurali olarak uygula.

Calistirma: PYTHONPATH=. python evaluation/analyze_keyword_overlap_signal.py
"""

from __future__ import annotations

import glob
import json

from app.rag.chunking import chunk_markdown_file
from app.rag.qdrant_retriever import QdrantRetriever
from app.rag.tfidf_retriever import tokenize


def load_chunks():
    chunks = []
    for path in sorted(glob.glob("data/policies/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks.extend(chunk_markdown_file(path, text))
    return chunks


def main() -> None:
    with open("evaluation/eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    chunks = load_chunks()
    retriever = QdrantRetriever(chunks, collection_name="thy_policies_ctx", recreate=False)

    rows = []
    for q in questions:
        results = retriever.search(q["question"], top_k=1)
        if not results:
            continue
        top_chunk, emb_score = results[0]

        query_tokens = set(tokenize(q["question"]))
        chunk_tokens = set(tokenize(top_chunk.text)) | set(tokenize(top_chunk.section_title))
        shared = query_tokens & chunk_tokens

        rows.append({
            "category": q["category"],
            "question": q["question"],
            "emb_score": emb_score,
            "shared_token_count": len(shared),
            "shared_tokens": sorted(shared),
        })

    print(f"{'Kategori':<18} {'Emb skor':>9} {'Ortak kelime #':>15}  Ornek ortak kelimeler")
    print("-" * 90)
    for r in rows:
        print(f"{r['category']:<18} {r['emb_score']:>9.3f} {r['shared_token_count']:>15}  {r['shared_tokens'][:5]}")

    print("\n=== KATEGORI BAZLI ORTALAMA ORTAK KELIME SAYISI ===")
    categories = sorted(set(r["category"] for r in rows))
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        avg_shared = sum(r["shared_token_count"] for r in cat_rows) / len(cat_rows)
        zero_shared_pct = sum(1 for r in cat_rows if r["shared_token_count"] == 0) / len(cat_rows)
        print(f"  {cat} (n={len(cat_rows)}): ortalama ortak kelime = {avg_shared:.2f}, "
              f"%{zero_shared_pct*100:.0f} soru hic ortak kelimesiz")

    with open("evaluation/keyword_overlap_analysis.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
