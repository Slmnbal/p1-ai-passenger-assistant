"""Deney: Hybrid search (BM25 + embedding, Reciprocal Rank Fusion) — ADR-0004
Yapılacaklar #3, ikinci madde.

Hipotez: Salt embedding araması, "kafa_karistirici" (kafa karıştırıcı/benzer) ve
"es_anlamli" (eş anlamlı) kategorilerinde zayıf (bkz. run_eval.py baseline: sırasıyla
Recall@1 %50 ve %20). Bu iki kategori genelde ANAHTAR KELİMENİN kendisinin (örn. "economy"
vs "business", "iç hat" vs "dış hat") doğru chunk'ı ayırt etmede belirleyici olduğu
durumlar — tam da BM25'in (kelime eşleşmesi) güçlü olduğu, embedding'in (anlamsal
benzerlik) bazen bu ayrımı bulanıklaştırdığı senaryo. Hibrit arama (BM25 + embedding,
Reciprocal Rank Fusion ile birleştirme) bu iki kategoride kazanım sağlayabilir.

Tek değişken: sıralama stratejisi (embedding-only vs hybrid). Embedding tarafı zaten
"thy_policies" (context-prefixed, kabul edilmiş varsayılan) collection'ını kullanır —
chunking/embedding modeli DEĞİŞMEDEN, üstüne bir BM25 katmanı eklenir.

Çalıştırma: PYTHONPATH=. python evaluation/hybrid_search_experiment.py
"""

from __future__ import annotations

import glob
import json
import re

from rank_bm25 import BM25Okapi

from app.rag.chunking import Chunk, chunk_markdown_file
from app.rag.qdrant_retriever import QdrantRetriever

_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+")
_CANDIDATE_POOL = 50  # her yöntemden RRF'ye giren aday sayısı
_RRF_K = 60  # standart RRF sabiti


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(glob.glob("data/policies/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks.extend(chunk_markdown_file(path, text))
    return chunks


class HybridRetriever:
    """Embedding (Qdrant, context-prefixed) + BM25 sonuçlarını RRF ile birleştirir."""

    def __init__(self, chunks: list[Chunk], embedding_retriever: QdrantRetriever):
        self.chunks = chunks
        self.embedding_retriever = embedding_retriever
        corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        # Embedding tarafi: mevcut Qdrant collection'indan genis bir aday havuzu
        emb_results = self.embedding_retriever.search(query, top_k=_CANDIDATE_POOL)
        emb_rank = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(emb_results)}

        # BM25 tarafi: ayni chunk listesi uzerinde skorla, en iyi N'i al
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:_CANDIDATE_POOL]
        bm25_rank = {self.chunks[i].chunk_id: rank for rank, i in enumerate(bm25_ranked_idx)}

        chunk_by_id = {c.chunk_id: c for c in self.chunks}
        all_ids = set(emb_rank) | set(bm25_rank)

        def rrf_score(chunk_id: str) -> float:
            score = 0.0
            if chunk_id in emb_rank:
                score += 1.0 / (_RRF_K + emb_rank[chunk_id] + 1)
            if chunk_id in bm25_rank:
                score += 1.0 / (_RRF_K + bm25_rank[chunk_id] + 1)
            return score

        fused = sorted(all_ids, key=lambda cid: -rrf_score(cid))[:top_k]
        return [(chunk_by_id[cid], rrf_score(cid)) for cid in fused]


def main() -> None:
    with open("evaluation/eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    chunks = _load_chunks()
    embedding_retriever = QdrantRetriever(chunks, collection_name="thy_policies", recreate=False)
    hybrid = HybridRetriever(chunks, embedding_retriever)

    results = []
    for q in questions:
        search_results = hybrid.search(q["question"], top_k=3)
        top_sources = [c.source_file for c, _ in search_results]
        expected = q.get("expected_source_file")
        if expected is not None:
            recall_1 = expected in top_sources[:1]
            recall_3 = expected in top_sources[:3]
        else:
            recall_1 = None
            recall_3 = None
        results.append({
            "id": q["id"], "question": q["question"], "category": q["category"],
            "expected_source_file": expected,
            "top1_source": top_sources[0] if top_sources else None,
            "recall_1": recall_1, "recall_3": recall_3,
        })

    with open("evaluation/eval_results_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    positive = [r for r in results if r["recall_1"] is not None]
    recall_1_rate = sum(1 for r in positive if r["recall_1"]) / len(positive)
    recall_3_rate = sum(1 for r in positive if r["recall_3"]) / len(positive)
    print(f"\n=== GENEL (hybrid, pozitif sorular, n={len(positive)}) ===")
    print(f"Recall@1: {recall_1_rate:.1%}")
    print(f"Recall@3: {recall_3_rate:.1%}")

    print("\n=== KATEGORI BAZLI (hybrid) ===")
    for cat in sorted(set(r["category"] for r in positive)):
        cat_results = [r for r in positive if r["category"] == cat]
        r1 = sum(1 for r in cat_results if r["recall_1"]) / len(cat_results)
        r3 = sum(1 for r in cat_results if r["recall_3"]) / len(cat_results)
        print(f"  {cat} (n={len(cat_results)}): Recall@1={r1:.1%}  Recall@3={r3:.1%}")


if __name__ == "__main__":
    main()
