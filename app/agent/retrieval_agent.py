"""Arama/RAG agent'ı — Adım 6, projenin İLK gerçek LLM cevap üretme adımı.

Adım 0-5 boyunca hiçbir yerde LLM'e serbest metin yazdırılmadı (retrieval sadece chunk
getirdi, intent sınıflandırma kapalı bir etiket kümesi döndürdü, tool'lar şablonlanmış
mesaj döndürdü). Burada ilk kez Ollama/llama3.1:8b'ye retrieved chunk'lara dayanarak
serbest bir cevap yazdırıyoruz — bu yüzden sistem promptu, modelin KAYNAK DIŞINA ÇIKMASINI
(halüsinasyon) engellemeye odaklı: yalnızca verilen bağlamı kullan, yoksa "bulamadım" de.

Bu, İlke 4'ün ("zorunlu kaynak atfı", "faithfulness kontrolü") TAM guardrail hali değil —
o, ayrı bir doğrulama adımı olarak Adım 7'de NLI/ikinci-LLM-çağrısıyla kurulacak. Burada
yapılan, prompt seviyesinde bir ÖN ÖNLEM; asıl kontrol policy_verification_agent'taki
temel groundedness kontrolüdür (bkz. o dosyanın docstring'i).

`get_retriever()` policy_lookup'tan paylaşılıyor (bkz. o dosyanın güncellenmiş docstring'i)
— embedding modelini ikinci kez yüklememek için.

**Adım 9 eklentisi:** `request_id` verilirse, bu LLM çağrısı Langfuse'a bir "generation"
olarak (model adı, prompt, tamamlanan metin, token kullanımı ile) kaydedilir — projenin
LLM/agent trace kanıtının somut bir parçası.
"""

from __future__ import annotations

import os

from openai import OpenAI

from app.agent.state import ConversationState
from app.observability import tracing
from app.tools.policy_lookup import get_retriever

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

_SYSTEM_PROMPT = (
    "Sen Turkish Airlines yolcu deneyimi asistanısın. Sana bir soru ve bu soruyla "
    "ilgili resmi kaynak metin parçaları (bağlam) verilecek.\n"
    "KURALLAR:\n"
    "1. SADECE verilen bağlamdaki bilgiyi kullan. Bağlamda olmayan hiçbir bilgiyi "
    "uydurma veya genel bilginle tamamlama.\n"
    "2. Bağlam soruyu yanıtlamaya yetmiyorsa, tam olarak şunu söyle: "
    "\"Bu konuda kaynaklarımda net bir bilgi bulamadım.\"\n"
    "3. Kısa ve net cevap ver, sayısal değerleri (kg, saat, gün, ücret) bağlamdaki "
    "haliyle birebir aktar.\n"
    "4. Kullanıcının sorduğu dilde (Türkçe veya İngilizce) cevap ver."
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "ollama-local-no-key-needed"),
        )
    return _client


def _build_context(sources: list[dict]) -> str:
    blocks = [
        f"[Kaynak: {s['source_file']} - {s['section_title']}]\n{s['text']}" for s in sources
    ]
    return "\n\n".join(blocks)


def retrieve_and_answer(state: ConversationState, top_k: int = 3) -> dict:
    question = state["user_message"]
    retriever = get_retriever()
    results = retriever.search(question, top_k=top_k)

    retrieved_sources = [
        {
            "source_file": chunk.source_file,
            "section_title": chunk.section_title,
            "score": float(score),
            "text": chunk.text,
        }
        for chunk, score in results
    ]

    context = _build_context(retrieved_sources)
    user_prompt = f"Bağlam:\n{context}\n\nSoru: {question}"
    client = _get_client()

    start_time = tracing.now()
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    end_time = tracing.now()
    answer = response.choices[0].message.content.strip()

    request_id = state.get("request_id")
    if request_id:
        usage = getattr(response, "usage", None)
        tracing.log_generation(
            request_id=request_id,
            name="retrieval_agent.answer_generation",
            model=OLLAMA_MODEL,
            start_time=start_time,
            end_time=end_time,
            input_data=user_prompt,
            output_data=answer,
            usage=(
                {
                    "promptTokens": usage.prompt_tokens,
                    "completionTokens": usage.completion_tokens,
                    "totalTokens": usage.total_tokens,
                }
                if usage
                else None
            ),
        )

    return {
        "retrieved_sources": retrieved_sources,
        "policy_answer": answer,
    }
