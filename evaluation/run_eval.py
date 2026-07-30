"""Adım 3 — etiketli değerlendirme seti üzerinde Recall@1 / Recall@3 ölçümü.

Bu script, evaluation/eval_questions.json'daki her soruyu QdrantRetriever ile arar ve
beklenen kaynak dosyanın (expected_source_file) top-1 ve top-3 sonuçların içinde olup
olmadığını kontrol eder. Kapsam dışı (negatif) sorular için beklenen dosya yoktur; bunun
yerine top-1 skoru raporlanır (İlke 4'teki güven eşiği tartışması için ham veri).

Çalıştırma: PYTHONPATH=. python evaluation/run_eval.py
Not gerektirir: Qdrant çalışıyor olmalı ve thy_policies collection'ı güncel olmalı
(evaluation/qdrant_migration_check.py ile yüklenmiş olmalı).
"""

from __future__ import annotations

import argparse
import json

from app.rag.qdrant_retriever import QdrantRetriever
from app.rag.chunking import chunk_markdown_file
import glob


def load_chunks():
    chunks = []
    for path in sorted(glob.glob("data/policies/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks.extend(chunk_markdown_file(path, text))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection", default="thy_policies",
        help="Olcum yapilacak Qdrant collection adi (baseline: thy_policies, deney: thy_policies_ctx)",
    )
    parser.add_argument(
        "--output", default="evaluation/eval_results.json",
        help="Sonuclarin yazilacagi dosya",
    )
    args = parser.parse_args()

    with open("evaluation/eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Toplam soru sayisi: {len(questions)}")
    print(f"Olcum yapilan collection: {args.collection}")

    chunks = load_chunks()
    retriever = QdrantRetriever(chunks, collection_name=args.collection, recreate=False)

    results = []
    for q in questions:
        search_results = retriever.search(q["question"], top_k=3)
        top_sources = [c.source_file for c, _ in search_results]
        top1_score = search_results[0][1] if search_results else 0.0

        expected = q.get("expected_source_file")
        if expected is not None:
            recall_1 = expected in top_sources[:1]
            recall_3 = expected in top_sources[:3]
        else:
            recall_1 = None
            recall_3 = None

        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "expected_source_file": expected,
            "top1_source": top_sources[0] if top_sources else None,
            "top1_score": top1_score,
            "recall_1": recall_1,
            "recall_3": recall_3,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Genel ozet (negatif/kapsam disi sorular haric)
    positive = [r for r in results if r["recall_1"] is not None]
    negative = [r for r in results if r["recall_1"] is None]

    recall_1_rate = sum(1 for r in positive if r["recall_1"]) / len(positive)
    recall_3_rate = sum(1 for r in positive if r["recall_3"]) / len(positive)

    print(f"\n=== GENEL (pozitif sorular, n={len(positive)}) ===")
    print(f"Recall@1: {recall_1_rate:.1%}")
    print(f"Recall@3: {recall_3_rate:.1%}")

    print(f"\n=== KATEGORI BAZLI ===")
    categories = sorted(set(r["category"] for r in positive))
    for cat in categories:
        cat_results = [r for r in positive if r["category"] == cat]
        r1 = sum(1 for r in cat_results if r["recall_1"]) / len(cat_results)
        r3 = sum(1 for r in cat_results if r["recall_3"]) / len(cat_results)
        print(f"  {cat} (n={len(cat_results)}): Recall@1={r1:.1%}  Recall@3={r3:.1%}")

    print(f"\n=== KAPSAM DISI SORULAR (n={len(negative)}) — top-1 skor dagilimi ===")
    for r in negative:
        print(f"  {r['top1_score']:.3f}  {r['question'][:60]}")
    avg_neg_score = sum(r["top1_score"] for r in negative) / len(negative)
    print(f"  Ortalama kapsam-disi top-1 skoru: {avg_neg_score:.3f}")


if __name__ == "__main__":
    main()
