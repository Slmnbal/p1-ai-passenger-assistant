"""Hugging Face sentence-transformers ile üretim (production) embedding retriever'ı.

Model seçimi: çok dilli `paraphrase-multilingual-MiniLM-L12-v2` kullanılıyor, tek
dilli/İngilizce `all-MiniLM-L6-v2` değil. Gerekçe (bkz. docs/adr/0001-embedding-model-secimi.md):
`evaluation/tfidf_vs_embedding_demo.py` ile yapılan 5 sorguluk ön-testte İngilizce model,
Türkçe eş anlamlı ifadelerde TF-IDF'ten daha iyi çıkmadı ve kapsam dışı bir soruya
(evcil hayvan) yanlışlıkla yüksek güven skoru (0.533) verdi — bu, güven eşiğine dayanan
güvenlik kontrolü (İlke 4) için tehlikeli. Çok dilli model aynı soruda çok daha düşük,
doğru bir skor (0.198) verdi ve iki ek soruda da doğru bölümü buldu. Bu hâlâ küçük
ölçekli (26 chunk, 5 soru) bir ön-test; Adım 3'teki 150-300 chunk / 50-100 soruluk
resmi değerlendirmede yeniden ölçülüp doğrulanacak.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.chunking import Chunk

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingRetriever:
    """Chunk listesi üzerinde HF sentence-transformers embeddingleriyle arama yapar."""

    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_MODEL_NAME):
        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        texts = [c.text for c in chunks]
        self._doc_embeddings = (
            self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            if texts
            else np.zeros((0, self.model.get_sentence_embedding_dimension()))
        )

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = self._doc_embeddings @ query_embedding
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
