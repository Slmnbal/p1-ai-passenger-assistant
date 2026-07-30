"""Qdrant tabanlı retriever — Adım 2'deki EmbeddingRetriever'ın kalıcı/ölçeklenebilir hali.

EmbeddingRetriever (embeddings.py), embedding'leri her çalıştırmada RAM'de yeniden
hesaplıyordu. Bu modül aynı embedding modelini kullanır ama vektörleri Qdrant'a
(Docker'da çalışan vektör veritabanı) yazar — kalıcı, filtrelenebilir ve büyük ölçekte
hızlı arama sağlar (bkz. p1_proje_plani.md Adım 3).

Varsayılan embed stratejisi "context-prefixed"tir: her chunk, embed edilmeden önce
belge başlığı + bölüm başlığıyla zenginleştirilir (bkz. docs/adr/0002-context-prefix-ve-
guven-esigi-siniri.md). Bu, ölçülmüş bir Recall@1 iyileştirmesi (%56→%72) sağladığı için
Adım 6+ için varsayılan yapıldı — çağıranın ayrıca bir şey yapmasına gerek yok.

**Adım 10 düzeltmesi:** `qdrant_url` artık `QDRANT_URL` ortam değişkeninden okunuyor
(varsayılan hâlâ `localhost:6333`, yerelde değişiklik yok). Neden gerekliydi:
container içinde çalışırken "localhost", app container'ının kendisini işaret eder,
`qdrant` servisini DEĞİL — `docker-compose.yml`'deki `app` servisi `QDRANT_URL=
http://qdrant:6333` ile bunu geçersiz kılıyor. Projedeki diğer tüm dış adresler
(OPENAI_BASE_URL, LANGFUSE_HOST) zaten bu desende; bu, gözden kaçmış tek istisnaydı.
"""

from __future__ import annotations

import os
import uuid
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.rag.chunking import Chunk
from app.rag.embeddings import DEFAULT_MODEL_NAME


def identity_embed_text(chunk: Chunk) -> str:
    """Eski/baseline davranış: yalnızca chunk metnini embed et (bkz. ADR-0002 karşılaştırması).

    Yeni kod bunu kullanmamalı — yalnızca ADR-0002'deki baseline ölçümü yeniden üretmek
    isteyenler için tutulur.
    """
    return chunk.text


def _infer_document_titles(chunks: list[Chunk]) -> dict[str, str]:
    """Her source_file için belge başlığını, o dosyanın ilk chunk'ının section_title'ından çıkarır.

    chunking.py'deki _split_into_sections, bir markdown dosyasının `# Başlık` satırını da
    bir header eşleşmesi saydığından, her dosyanın ilk chunk'ının section_title'ı zaten o
    belgenin başlığıdır — dosyaları tekrar okumaya gerek yok.
    """
    titles: dict[str, str] = {}
    for chunk in chunks:
        titles.setdefault(chunk.source_file, chunk.section_title)
    return titles


def build_context_prefixed_embed_fn(chunks: list[Chunk]) -> Callable[[Chunk], str]:
    """ADR-0002'de kabul edilen varsayılan strateji: belge başlığı + bölüm başlığı + metin."""
    doc_titles = _infer_document_titles(chunks)

    def _embed_text(chunk: Chunk) -> str:
        doc_title = doc_titles.get(chunk.source_file, chunk.source_file)
        return f"{doc_title}. {chunk.section_title}. {chunk.text}"

    return _embed_text


class QdrantRetriever:
    """Chunk listesini Qdrant'a yükler ve üzerinde anlamsal arama yapar.

    `embed_text_fn`: embedding'e giren metni oluşturan fonksiyon. Chunk'ın kendisi
    (payload'daki `text`) her zaman orijinal kalır — yalnızca aranırken kullanılan vektör
    farklı bir metinden hesaplanabilir. Belirtilmezse ADR-0002'nin kabul ettiği
    context-prefixed strateji kullanılır.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        collection_name: str = "thy_policies",
        qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333"),
        model_name: str = DEFAULT_MODEL_NAME,
        recreate: bool = True,
        embed_text_fn: Callable[[Chunk], str] | None = None,
    ):
        self.chunks = chunks
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.model = SentenceTransformer(model_name)
        self.embed_text_fn = embed_text_fn or build_context_prefixed_embed_fn(chunks)

        if recreate:
            self._create_collection()
            self._upsert_chunks()

    def _create_collection(self) -> None:
        vector_size = self.model.get_sentence_embedding_dimension()
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def _upsert_chunks(self) -> None:
        if not self.chunks:
            return
        texts = [self.embed_text_fn(c) for c in self.chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "section_title": chunk.section_title,
                    "text": chunk.text,
                },
            )
            for i, chunk in enumerate(self.chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
        )
        output = []
        for point in result.points:
            payload = point.payload
            chunk = Chunk(
                chunk_id=payload["chunk_id"],
                source_file=payload["source_file"],
                section_title=payload["section_title"],
                text=payload["text"],
            )
            output.append((chunk, point.score))
        return output
