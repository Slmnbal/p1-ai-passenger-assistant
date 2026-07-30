"""TF-IDF tabanlı baseline retriever.

RAG'de de "önce basit baseline, sonra gelişmiş model" ilkesini uyguluyoruz (bkz.
p1_proje_plani.md — Data Scientist yaklaşımı, Adım 4'teki intent router için de aynı
desen kullanıldı). Bu modül hiçbir harici bağımlılık gerektirmez (yalnızca stdlib +
numpy), bu yüzden internet erişimi olmayan ortamlarda bile test edilebilir.

embeddings.py'deki sentence-transformers tabanlı EmbeddingRetriever ile Adım 3'te
Recall@k / MRR üzerinden karşılaştırılacak; TF-IDF'in nerede yetersiz kaldığını
(eş anlamlı kelimeler, anlamsal benzerlik) somut örneklerle göstermek başlı başına
bir portföy kanıtıdır.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from app.rag.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", re.UNICODE)

# Türkçe + İngilizce çok sık geçen, ayırt edici olmayan kelimeler
_STOPWORDS = {
    "ve", "ile", "bir", "bu", "da", "de", "için", "veya", "olan", "olarak", "gibi",
    "her", "en", "ya", "kadar", "sonra", "önce", "olur", "olan", "olarak", "ise",
    "the", "and", "or", "of", "to", "in", "a", "is", "for", "on", "at", "be",
}


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


class TfidfRetriever:
    """Chunk listesi üzerinde TF-IDF + kosinüs benzerliği ile arama yapan baseline."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._doc_vectors: np.ndarray | None = None
        self._build_index()

    def _build_index(self) -> None:
        tokenized_docs = [tokenize(c.text) for c in self.chunks]

        vocab: dict[str, int] = {}
        for tokens in tokenized_docs:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self._vocab = vocab

        n_docs = len(self.chunks)
        n_vocab = max(len(vocab), 1)

        doc_freq = np.zeros(n_vocab)
        tf_matrix = np.zeros((n_docs, n_vocab))

        for doc_idx, tokens in enumerate(tokenized_docs):
            counts = Counter(tokens)
            total = sum(counts.values()) or 1
            for tok, count in counts.items():
                col = vocab[tok]
                tf_matrix[doc_idx, col] = count / total
            for tok in set(tokens):
                doc_freq[vocab[tok]] += 1

        idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1  # smooth idf (sklearn ile aynı formül)
        self._idf = idf
        self._doc_vectors = tf_matrix * idf

        norms = np.linalg.norm(self._doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._doc_vectors = self._doc_vectors / norms

    def _vectorize_query(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        vec = np.zeros(len(self._vocab) or 1)
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        for tok, count in counts.items():
            if tok in self._vocab:
                col = self._vocab[tok]
                vec[col] = (count / total) * self._idf[col]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_vec = self._vectorize_query(query)
        scores = self._doc_vectors @ query_vec
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
