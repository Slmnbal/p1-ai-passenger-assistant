"""Intent veri seti EDA'si: dagilim, dil karisimi, mesaj uzunlugu, sinif-ici kelime ortusmesi.

Calistirma: PYTHONPATH=. python data/intent/eda.py
Cikti: data/intent/eda_results.json + konsola ozet
"""

from __future__ import annotations

import json
from collections import Counter

from app.rag.tfidf_retriever import tokenize


def main() -> None:
    with open("data/intent/messages.json", encoding="utf-8") as f:
        records = json.load(f)

    intents = sorted(set(r["intent"] for r in records))

    print(f"Toplam mesaj: {len(records)}\n")

    print("=== INTENT DAGILIMI ===")
    dist = Counter(r["intent"] for r in records)
    for intent in intents:
        pct = dist[intent] / len(records) * 100
        print(f"  {intent}: {dist[intent]} (%{pct:.1f})")

    print("\n=== DIL KARISIMI (genel) ===")
    lang_dist = Counter(r["language"] for r in records)
    for lang, count in lang_dist.most_common():
        print(f"  {lang}: {count} (%{count/len(records)*100:.1f})")

    print("\n=== SINIF BAZLI DIL KARISIMI ===")
    for intent in intents:
        cls_records = [r for r in records if r["intent"] == intent]
        cls_lang = Counter(r["language"] for r in cls_records)
        print(f"  {intent}: {dict(cls_lang)}")

    print("\n=== MESAJ UZUNLUGU (kelime sayisi) ===")
    for intent in intents:
        lens = [r["word_len"] for r in records if r["intent"] == intent]
        avg = sum(lens) / len(lens)
        print(f"  {intent}: ort={avg:.1f}, min={min(lens)}, max={max(lens)}")

    overall_lens = [r["word_len"] for r in records]
    print(f"  GENEL: ort={sum(overall_lens)/len(overall_lens):.1f}, "
          f"min={min(overall_lens)}, max={max(overall_lens)}")

    print("\n=== SINIF-ICI KELIME ORTUSMESI (Jaccard benzerligi, ortalama) ===")
    class_vocabs = {}
    for intent in intents:
        cls_records = [r for r in records if r["intent"] == intent]
        vocab = set()
        for r in cls_records:
            vocab.update(tokenize(r["text"]))
        class_vocabs[intent] = vocab

    overlap_matrix = {}
    for i1 in intents:
        overlap_matrix[i1] = {}
        for i2 in intents:
            if i1 == i2:
                continue
            v1, v2 = class_vocabs[i1], class_vocabs[i2]
            jaccard = len(v1 & v2) / len(v1 | v2) if (v1 | v2) else 0.0
            overlap_matrix[i1][i2] = round(jaccard, 3)

    for i1 in intents:
        sorted_overlaps = sorted(overlap_matrix[i1].items(), key=lambda x: -x[1])
        top = sorted_overlaps[0]
        print(f"  {i1}: en cok ortustugu sinif = {top[0]} (Jaccard={top[1]})")

    results = {
        "total": len(records),
        "distribution": dict(dist),
        "language_distribution": dict(lang_dist),
        "class_language": {intent: dict(Counter(r["language"] for r in records if r["intent"] == intent)) for intent in intents},
        "length_stats": {
            intent: {
                "avg_words": sum(r["word_len"] for r in records if r["intent"] == intent) / dist[intent],
                "min_words": min(r["word_len"] for r in records if r["intent"] == intent),
                "max_words": max(r["word_len"] for r in records if r["intent"] == intent),
            }
            for intent in intents
        },
        "vocab_overlap_jaccard": overlap_matrix,
    }

    with open("data/intent/eda_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
