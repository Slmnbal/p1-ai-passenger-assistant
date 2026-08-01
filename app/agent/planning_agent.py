"""Planlama agent'ı — Adım 6'nın giriş node'u.

Adım 4'te secilen "full fine-tune 10 epoch" modelini (`models/intent_full_10ep`,
bkz. `data/intent/MODEL_CARD.md`) kullanarak mesaji 6 intent'ten birine atar, sonra
akisi buna gore yonlendirir:
- `kapsam_disi` -> bu node dogrudan `final_response` uretir, graph burada biter.
- `politika_bilgi_sorgusu` / `belirsiz_acikliga_kavusturma` -> retrieval_agent

**1 Ağustos 2026 güncellemesi (bkz. docs/adr/0006-...md):** `belirsiz_acikliga_
kavusturma` artık DOĞRUDAN bitmiyor, RAG'e de bir şans veriyor — canlı testte
"Kayıp bagajım için ne yapmalıyım?" gibi AÇIK politika soruları bile bu sınıfa
düşebiliyordu (intent modelinin ölçülen ~%24 hata oranının bir parçası, bkz.
MODEL_CARD.md) ve RAG'e hiç ulaşamıyordu. Adım 6'nın zaten kanıtladığı prensip
("RAG'in kendi groundedness kontrolü, intent'in kaçırdığı bir soruyu yakalayabilir")
burada tersine de uygulandı: RAG grounded bir cevap bulursa onu döndürür,
bulamazsa (bkz. `policy_verification_agent.py::verify_node`) mevcut netleştirme
mesajına düşer — kullanıcı hiçbir durumda daha kötü bir deneyim yaşamaz, sadece
ekstra bir RAG denemesi (gecikme maliyeti) eklenir.
- `ucus_sorgulama` / `rezervasyon_islem_talebi` / `checkin_talebi` -> tool_agent

Guven skoru NEDEN akis kararinda KULLANILMIYOR: MODEL_CARD.md'deki kalibrasyon
olcumu (0.7-0.9 guven araliginda gercek dogruluk sadece %53.3) burada yeniden
dogrulandi — `eval_results.json` uzerinde farkli esikler denendiginde (0.3'ten 0.8'e)
tutma orani ile dogruluk arasinda net/monoton bir iliski yok (bkz. proje hafizasi
"professional-rigor-mentality": olcmeden esik uydurmadik). Bu yuzden `intent_confidence`
state'e sadece GOZLEMLENEBILIRLIK icin (Adim 9 audit log) yaziliyor; akis karari
dogrudan siniflandiricinin tahmin ettigi sinifa dayaniyor — `belirsiz_acikliga_kavusturma`
zaten kendi basina bir sinif oldugu icin dusuk-guven durumu byle dolayli olarak ele
alinmis oluyor.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.agent.state import ConversationState
from app.intent.rule_based_router import INTENTS

MODEL_DIR = "models/intent_full_10ep"
ID2LABEL = dict(enumerate(INTENTS))

_KAPSAM_DISI_MESSAGE = (
    "Bu konu yolcu deneyimi asistanının kapsamı dışında görünüyor (kargo, işe alım, "
    "yatırımcı ilişkileri gibi konularda yardımcı olamıyorum). Turkish Airlines'ın ilgili "
    "kurumsal kanallarına yönlenmenizi öneririm."
)
_BELIRSIZ_MESSAGE = (
    "Sorunuzu tam olarak anlayamadım. Biraz daha detaylandırır mısınız? Örneğin: "
    "uçuşunuzun durumuyla mı, bir rezervasyon işlemiyle mi (iptal/tarih değişikliği/"
    "bagaj), check-in ile mi, yoksa genel bir politika/kural bilgisiyle mi ilgili?"
)

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
    return _tokenizer, _model


def classify_intent(text: str) -> tuple[str, float]:
    tokenizer, model = _get_model()
    inputs = tokenizer(text, truncation=True, padding=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    return ID2LABEL[pred_id], float(probs[pred_id].item())


def plan_node(state: ConversationState) -> dict:
    intent, confidence = classify_intent(state["user_message"])
    update: dict = {
        "intent": intent,
        "intent_confidence": confidence,
        "history": [{"role": "user", "content": state["user_message"]}],
    }

    if intent == "kapsam_disi":
        update["final_response"] = _KAPSAM_DISI_MESSAGE
    elif intent == "belirsiz_acikliga_kavusturma":
        update["needs_clarification"] = True
        update["clarification_question"] = _BELIRSIZ_MESSAGE
        update["final_response"] = _BELIRSIZ_MESSAGE

    return update


def route_after_planning(state: ConversationState) -> str:
    intent = state["intent"]
    if intent == "kapsam_disi":
        return "end"
    if intent in ("politika_bilgi_sorgusu", "belirsiz_acikliga_kavusturma"):
        return "retrieval"
    return "tools"
