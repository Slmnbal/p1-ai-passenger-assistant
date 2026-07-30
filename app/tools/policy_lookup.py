"""Politika sorgulama araci — RAG (QdrantRetriever) katmanina kopru.

Diger 3 tool (flight_search, reservation, checkin) basit fonksiyonlardir, ama RAG
retrieval'i cagirmak icin once bir QdrantRetriever NESNESI olusturmak gerekir — bu,
embedding modelini (sentence-transformers, ~470MB) yuklemek demektir ve birkac saniye
surer. Bunu HER SORGUDA tekrar yapmak cok yavas olurdu.

Bu yuzden retriever, modul ilk import edildiginde BIR KERE olusturulur (asagidaki
`get_retriever()` fonksiyonu, sonuc bir kere hesaplanip _retriever_instance'da
saklanir — "lazy singleton" deseni). Boylece LangGraph (Adim 6) icin diger tool'larla
ayni basit fonksiyon arayuzu ("bir soru ver, cevap al") korunurken, pahali model yukleme
islemi yalnizca bir kere yapilir.

`get_retriever()` bilinçli olarak public: Adım 6'nın `retrieval_agent.py`'ı, chunk'ların
tam metnine (`PolicyQueryResponse` sadece kaynak/skor dondurur, metin degil) ihtiyac
duyuyor ve embedding modelini IKINCI KEZ yuklememesi icin ayni singleton'i paylasiyor.

recreate=False onemli: Adim 3'te Qdrant'a zaten yuklenmis olan "thy_policies"
collection'ini yeniden olusturmaya CALISMAZ, sadece var olana baglanir. Bu, uygulama
her baslatildiginda 302 chunk'in yeniden embed edilip Qdrant'a yazilmasini onler.
"""

from __future__ import annotations

import glob

from app.rag.chunking import chunk_markdown_file
from app.rag.qdrant_retriever import QdrantRetriever
from app.tools.schemas import PolicyQueryRequest, PolicyQueryResponse, PolicySource

_retriever_instance: QdrantRetriever | None = None


def _load_chunks():
    chunks = []
    for path in sorted(glob.glob("data/policies/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks.extend(chunk_markdown_file(path, text))
    return chunks


def get_retriever() -> QdrantRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        chunks = _load_chunks()
        _retriever_instance = QdrantRetriever(chunks, recreate=False)
    return _retriever_instance


def query_policy(request: PolicyQueryRequest, top_k: int = 3) -> PolicyQueryResponse:
    retriever = get_retriever()
    results = retriever.search(request.question, top_k=top_k)
    sources = [
        PolicySource(source_file=chunk.source_file, section_title=chunk.section_title, score=score)
        for chunk, score in results
    ]
    return PolicyQueryResponse(question=request.question, top_sources=sources)
