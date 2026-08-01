"""Politika doğrulama agent'ı — Adım 6, guardrail'e devretmeden ÖN kontrol.

Burada yapılan İKİ kontrol de bilinçli olarak BASİT/sezgisel — tam guardrail (NLI tabanlı
faithfulness, ikinci LLM çağrısıyla groundedness) Adım 7'nin işi. Bunu Adım 6'da tam
çözülmüş gibi göstermek "professional-rigor-mentality" ilkesine aykırı olurdu:

1. **Groundedness (temel):** Üretilen cevaptaki anlamlı kelimelerin ne kadarı retrieved
   chunk metinlerinde de geçiyor — basit bir kelime-örtüşme oranı. ADR-0002'nin (Adım 3)
   ölçtüğü gibi, kelime örtüşmesi kapsam-dışı/eş-anlamlı ayrımında güvenilir bir sinyal
   DEĞİL; ama burada farklı bir şey ölçüyoruz (cevap chunk'tan mı üretildi, chunk soruyla
   mı alakalı değil) ve LLM zaten kaynağa sadık kalması için promptlandı (bkz.
   retrieval_agent) — bu kontrol yalnızca ikinci bir sezgisel emniyet katmanı.

2. **Çakışan kaynak / netleştirme:** En iyi iki sonucun skoru birbirine çok yakınsa VE
   FARKLI KAYNAK DOSYADAN geliyorsa (örn. iki ayrı politika belgesi ters kural içeriyorsa),
   hangi bağlamın geçerli olduğu belirsiz demektir — cevap yerine netleştirici soru
   döndürülür (bkz. proje planı İlke 4). Bu da kesin bir çözüm değil, bilinen bir sezgisel
   eşik.

   **31 Temmuz 2026 düzeltmesi (bkz. docs/adr/0005-...md):** Eskiden bu kontrol yalnızca
   "farklı bölüm başlığı" bakıyordu — AYNI belgenin iki farklı (ama alakasız/gürültü)
   alt-bölümünü de "çakışma" sanıyordu (ADR-0004'ün e011 senaryosunda somut olarak
   ölçülen bir yanlış pozitif: İngilizce bir soruda aynı dosyanın iki ilgisiz alt-bölümü
   skor olarak yakın çıktı, sistem gereksiz yere netleştirici soru sordu). Artık "farklı
   BÖLÜM" yerine "farklı KAYNAK DOSYA" şartı aranıyor — gerçek bir politika çakışması,
   tanım gereği iki AYRI belge arasında olur, aynı belgenin iki alt-başlığı arasında
   değil.

`route_after_verification`, graph.py'nin retry/reflection kararını vermesi için "end"
veya "retry" döndürür — en fazla 1 retry (bkz. state.py'deki `retry_count`), sonsuz
döngüyü önlemek için.

**1 Ağustos 2026 (bkz. docs/adr/0006-...md):** `belirsiz_acikliga_kavusturma` artık
`planning_agent`'ta doğrudan bitmiyor, buraya da uğruyor (bkz. `route_after_planning`).
Bu yüzden `verify_node`'un fallback/grounded-değil dallarında, orijinal intent
`belirsiz_acikliga_kavusturma` ise generik "bulamadım" yerine `plan_node`'un zaten
ürettiği netleştirme sorusuna (`state["clarification_question"]`) dönülüyor.
"""

from __future__ import annotations

import re

from app.agent.state import ConversationState

_FALLBACK_PHRASE = "bu konuda kaynaklarımda net bir bilgi bulamadım"
_GROUNDEDNESS_THRESHOLD = 0.4
_SCORE_GAP_FOR_CONFUSION = 0.03
_MAX_RETRIES = 1


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zçğıöşü0-9]+", text.lower())
    return {w for w in words if len(w) > 3}


def _groundedness_ratio(answer: str, sources: list[dict]) -> float:
    context_words = set()
    for s in sources:
        context_words |= _content_words(s["text"])
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


def _sources_conflict(sources: list[dict]) -> bool:
    if len(sources) < 2:
        return False
    top1, top2 = sources[0], sources[1]
    close_scores = abs(top1["score"] - top2["score"]) < _SCORE_GAP_FOR_CONFUSION
    different_file = top1["source_file"] != top2["source_file"]
    return close_scores and different_file


def _format_sources(sources: list[dict]) -> str:
    lines = {f"- {s['source_file']} ({s['section_title']})" for s in sources}
    return "\n".join(sorted(lines))


def verify_node(state: ConversationState) -> dict:
    answer = state["policy_answer"]
    sources = state["retrieved_sources"]
    retry_count = state.get("retry_count", 0)

    if _FALLBACK_PHRASE in answer.lower():
        if state.get("intent") == "belirsiz_acikliga_kavusturma":
            question = state.get("clarification_question") or answer
            return {"grounded": False, "needs_clarification": True, "final_response": question}
        return {"grounded": False, "needs_clarification": False, "final_response": answer}

    if _sources_conflict(sources):
        question = (
            "Sorunuz birden fazla bağlama uyabiliyor (örn. farklı kabin sınıfı/hat türü "
            "kuralları) ve kaynaklarım bu konuda net bir ayrım yapamadı. Hangi durumu "
            "kastettiğinizi (kabin sınıfı, iç hat/dış hat gibi) belirtir misiniz?\n\n"
            f"Bulduğum ilgili kaynaklar:\n{_format_sources(sources)}"
        )
        return {
            "grounded": False,
            "needs_clarification": True,
            "clarification_question": question,
            "final_response": question,
        }

    ratio = _groundedness_ratio(answer, sources)
    grounded = ratio >= _GROUNDEDNESS_THRESHOLD

    if grounded:
        final = f"{answer}\n\nKaynak:\n{_format_sources(sources)}"
        return {
            "grounded": True,
            "needs_clarification": False,
            "final_response": final,
            "history": [{"role": "assistant", "content": final}],
        }

    if retry_count < _MAX_RETRIES:
        return {"grounded": False, "retry_count": retry_count + 1}

    # 1 Agustos 2026 (bkz. docs/adr/0006-...md): belirsiz_acikliga_kavusturma artik
    # RAG'e de ugruyor (bkz. planning_agent::route_after_planning). RAG grounded bir
    # cevap BULAMAZSA, generik "bulamadim" yerine plan_node'un zaten urettigi
    # netlestirme sorusuna donuyoruz — cunku gercekten belirsiz bir mesaj icin
    # "bulamadim" degil "ne demek istediginizi acar misiniz" daha dogru bir yanit.
    if state.get("intent") == "belirsiz_acikliga_kavusturma":
        question = state.get("clarification_question") or "Sorunuzu biraz daha detaylandırır mısınız?"
        return {"grounded": False, "needs_clarification": True, "final_response": question}

    fallback = "Bu konuda kaynaklarımda net bir bilgi bulamadım."
    return {"grounded": False, "final_response": fallback}


def route_after_verification(state: ConversationState) -> str:
    if state.get("final_response") is not None:
        return "end"
    return "retry"
