"""Rezervasyon/tool agent'ı — Adım 6.

planning_agent'ın yönlendirdiği `ucus_sorgulama`, `rezervasyon_islem_talebi` ve
`checkin_talebi` intent'lerini, Adım 5'in mock tool fonksiyonlarına (`app/tools/*.py`)
DOĞRUDAN Python çağrısıyla bağlar — `app/main.py`'daki docstring'in belirttiği gibi
tool fonksiyonları HTTP'den habersizdir, LangGraph bunları doğrudan çağırır, main.py'ın
HTTP sarmalayıcısı burada devre dışıdır.

Bilinçli tasarım kararı — final_response burada LLM ile ÜRETİLMİYOR: her tool fonksiyonu
zaten kendi şablonlanmış `message` alanını döndürüyor (örn. "İptal edildi, tahmini iade
X TRY"). Bu sayı/tarih içeren mesajı bir LLM'e yeniden yazdırmak, İlke 4'ün henüz kod
olarak yazılmamış ama tam da önlemeye çalıştığı riski (paraphrase kaynaklı sayı hatası)
gereksiz yere buraya taşırdı. LLM tabanlı cevap üretimi bilinçli olarak yalnızca
retrieval_agent'ta (serbest metin gerektiren politika soruları için) var.

Entity çıkarımı (PNR, uçuş no, tarih, bagaj adedi) regex/kural tabanlı — Adım 4'teki
"önce kural tabanlı baseline" desenine tutarlı (bkz. rule_based_router.py). Eksik/
belirsiz bir entity varsa (örn. PNR verilmemiş) tahmin YÜRÜTÜLMEZ, netleştirici soru
döndürülür (bkz. proje planı İlke 4, "belirsiz sorularda netleştirme").

**Adım 8 güncellemesi:** `cancel` ve `change_date` artık HEMEN çalıştırılmıyor —
`requires_human_approval` iş kuralına göre (bkz. `reservation.py`) bu ikisi kritik kabul
edildiği için `approval_queue.submit()` ile bekletiliyor, gerçek mutasyon yalnızca
`approval_flow.approve()` çağrıldığında olur. `add_baggage` kritik sayılmadığı için
(Adım 5'teki karar) değişmedi, hâlâ hemen çalışıyor.

**Adım 9 eklentisi:** Bu node'da gerçekten yürütülen (check-in, ekstra bagaj) veya
onaya gönderilen (iptal, tarih değişikliği) her işlem `audit_log.record()` ile
`actor="system"` olarak kaydediliyor — bkz. `approval_flow.py`'nin `actor="human_approver"`
kaydıyla arasındaki ayrım (kimin tetiklediği net olsun diye).
"""

from __future__ import annotations

import re

from app.agent.state import ConversationState
from app.human_in_the_loop import approval_queue
from app.observability import audit_log
from app.tools import checkin, flight_search, reservation, store

# Adım 11'de bulunan gerçek bug: eski kalıp `\b[A-Z]{3}[A-Z0-9]{4}\b` (yalnızca "3 harf +
# 4 alfanümerik") sıradan 7 harfli Türkçe kelimeleri de eşliyordu — "KAPANIR", "YOLCUYA"
# gibi kelimeler (mesaj .upper() ile büyütüldüğü için) yanlışlıkla PNR sanılıyordu (bkz.
# evaluation/e2e_scenarios.json e004/e006, docs/adr/0004). Bu mock sistemdeki TÜM PNR'ler
# "SYN" ile başladığı için (bkz. data/mock_reservations.json), kalıp bu önekle sınırlandı.
_PNR_RE = re.compile(r"\bSYN[A-Z0-9]{4}\b")
_FLIGHT_TK_RE = re.compile(r"\bTK\s?-?(\d{1,4})\b", re.IGNORECASE)
_FLIGHT_THY_RE = re.compile(r"\bTHY\s?-?(\d{1,4})\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_BAGGAGE_COUNT_RE = re.compile(r"\b([1-5])\b")

_CANCEL_KEYWORDS = ["iptal", "cancel", "vazgeç"]
_CHANGE_DATE_KEYWORDS = ["tarih", "erteleyebilir", "ertele", "değiştir", "change my", "reschedule"]
_ADD_BAGGAGE_KEYWORDS = ["bagaj", "baggage", "valiz", "çanta"]


def _extract_pnr(text: str) -> str | None:
    match = _PNR_RE.search(text.upper())
    return match.group(0) if match else None


def _extract_flight_no(text: str) -> str | None:
    m = _FLIGHT_TK_RE.search(text)
    if m:
        return f"TK{m.group(1)}"
    m = _FLIGHT_THY_RE.search(text)
    if m:
        return f"TK{m.group(1)}"
    return None


def _extract_date(text: str) -> str | None:
    match = _DATE_RE.search(text)
    return match.group(1) if match else None


def _extract_airport_codes(text: str) -> tuple[str, str] | None:
    known_codes = {f["origin"] for f in store.FLIGHTS} | {f["destination"] for f in store.FLIGHTS}
    found = [tok for tok in re.findall(r"\b[A-Z]{3}\b", text.upper()) if tok in known_codes]
    if len(found) >= 2:
        return found[0], found[1]
    return None


def _classify_reservation_action(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in _CANCEL_KEYWORDS):
        return "cancel"
    if any(kw in t for kw in _CHANGE_DATE_KEYWORDS):
        return "change_date"
    if any(kw in t for kw in _ADD_BAGGAGE_KEYWORDS):
        return "add_baggage"
    return "unknown"


def _needs_clarification(question: str) -> dict:
    return {
        "needs_clarification": True,
        "clarification_question": question,
        "final_response": question,
    }


def _finish(tool_result: dict, message: str) -> dict:
    return {
        "tool_result": tool_result,
        "final_response": message,
        "history": [{"role": "assistant", "content": message}],
    }


def tool_node(state: ConversationState) -> dict:
    text = state["user_message"]
    intent = state["intent"]
    request_id = state.get("request_id")
    entities = {
        k: v
        for k, v in {
            "pnr": _extract_pnr(text),
            "flight_no": _extract_flight_no(text),
            "date": _extract_date(text),
        }.items()
        if v is not None
    }

    if intent == "ucus_sorgulama":
        if entities.get("flight_no"):
            info = flight_search.get_flight_status(entities["flight_no"])
            if info is None:
                return _finish(
                    {"found": False},
                    f"{entities['flight_no']} numaralı bir uçuş bulamadım. Uçuş numarasını kontrol edebilir misiniz?",
                )
            status_tr = {"on_time": "zamanında", "delayed": "gecikmeli", "cancelled": "iptal edildi"}[info.status]
            msg = f"{info.flight_no} ({info.origin}→{info.destination}) uçuşu {status_tr}."
            if info.status == "delayed":
                msg += f" Gecikme: {info.delay_minutes} dakika."
            if info.gate:
                msg += f" Kapı: {info.gate}."
            return _finish({"flight_info": info.model_dump()}, msg)

        route = _extract_airport_codes(text)
        if route:
            origin, destination = route
            from app.tools.schemas import FlightSearchRequest

            resp = flight_search.search_flights(FlightSearchRequest(origin=origin, destination=destination))
            if not resp.flights:
                return _finish({"flights": []}, f"{origin}→{destination} arasında eşleşen bir uçuş bulamadım.")
            lines = [f"- {f.flight_no} ({f.typical_duration_min} dk, durum: {f.status})" for f in resp.flights]
            msg = f"{origin}→{destination} arası uçuşlar:\n" + "\n".join(lines)
            return _finish({"flights": [f.model_dump() for f in resp.flights]}, msg)

        return _needs_clarification(
            "Hangi uçuşu kastettiğinizi anlayamadım. Uçuş numaranızı (örn. TK2110) veya "
            "kalkış-varış havalimanı kodlarını paylaşır mısınız?"
        )

    if intent == "checkin_talebi":
        if not entities.get("pnr"):
            return _needs_clarification("Check-in yapabilmem için PNR numaranızı paylaşır mısınız?")
        result = checkin.check_in(entities["pnr"])
        if result is None:
            return _finish({"found": False}, f"{entities['pnr']} PNR'ı ile bir rezervasyon bulamadım.")
        audit_log.record(
            actor="system", action="checkin", request_id=request_id,
            pnr=entities["pnr"], status=result.status,
        )
        return _finish({"checkin": result.model_dump()}, result.message)

    if intent == "rezervasyon_islem_talebi":
        if not entities.get("pnr"):
            return _needs_clarification("Bu işlem için PNR numaranızı paylaşır mısınız?")

        action = _classify_reservation_action(text)

        if action == "cancel":
            if reservation.get_reservation(entities["pnr"]) is None:
                return _finish({"found": False}, f"{entities['pnr']} PNR'ı ile bir rezervasyon bulamadım.")
            request = approval_queue.submit("cancel", entities["pnr"], {})
            audit_log.record(
                actor="system", action="submit_approval_cancel", request_id=request_id,
                approval_request_id=request.id, pnr=entities["pnr"],
            )
            message = (
                f"İptal talebiniz alındı ve insan onayına gönderildi (talep no: {request.id}). "
                "Onaylandığında rezervasyonunuz iptal edilecek."
            )
            return _finish({"approval_request": request.model_dump()}, message)

        if action == "change_date":
            if not entities.get("date"):
                return _needs_clarification(
                    "Yeni uçuş tarihini YYYY-AA-GG formatında (örn. 2026-08-15) paylaşır mısınız?"
                )
            if reservation.get_reservation(entities["pnr"]) is None:
                return _finish({"found": False}, f"{entities['pnr']} PNR'ı ile bir rezervasyon bulamadım.")
            request = approval_queue.submit("change_date", entities["pnr"], {"new_date": entities["date"]})
            audit_log.record(
                actor="system", action="submit_approval_change_date", request_id=request_id,
                approval_request_id=request.id, pnr=entities["pnr"], new_date=entities["date"],
            )
            message = (
                f"Tarih değişikliği talebiniz alındı ve insan onayına gönderildi (talep no: {request.id})."
            )
            return _finish({"approval_request": request.model_dump()}, message)

        if action == "add_baggage":
            count_match = _BAGGAGE_COUNT_RE.search(text)
            if not count_match:
                return _needs_clarification("Kaç parça ekstra bagaj eklemek istiyorsunuz (1-5)?")
            result = reservation.add_baggage(entities["pnr"], int(count_match.group(1)))
            if result is None:
                return _finish({"found": False}, f"{entities['pnr']} PNR'ı ile bir rezervasyon bulamadım.")
            audit_log.record(
                actor="system", action="add_baggage", request_id=request_id,
                pnr=entities["pnr"], extra_pieces=int(count_match.group(1)),
            )
            return _finish({"add_baggage": result.model_dump()}, result.message)

        return _needs_clarification(
            "Rezervasyonunuzda ne yapmak istediğinizi tam anlayamadım — iptal mi, tarih "
            "değişikliği mi, yoksa ekstra bagaj eklemek mi istiyorsunuz?"
        )

    raise ValueError(f"tool_node bu intent'i işlemiyor: {intent}")
