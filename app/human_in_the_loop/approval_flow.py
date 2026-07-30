"""Onay/red simülasyonu — Adım 8.

`approve()`, ERTELENMİŞ olan gerçek tool çağrısını (`reservation.cancel_reservation` /
`change_flight_date`) BURADA, onay anında çalıştırır — Adım 5-6'da `tool_agent`'ın
mesaj gelir gelmez yaptığı mutasyon artık SADECE bu fonksiyon çağrıldığında gerçekleşir.
Bu, planın istediği "sonucun agent akışına geri beslenmesi": onaylanan işlemin gerçek
sonucu (iade tutarı, yeni tarih vb.) `request.result`'a yazılır; aynı PNR daha sonra
`reservation.get_reservation()` ile sorgulanırsa güncel hâli görülür.

`reject()` store'da HİÇBİR ŞEYİ değiştirmez — bu, onay mekanizmasının gerçek bir
denetim noktası olduğunun (yalnızca dekoratif değil) somut kanıtı.

**Adım 9 eklentisi:** Her `approve()`/`reject()` çağrısı `audit_log.record()` ile
`actor="human_approver"` olarak kaydediliyor — bu, "kim, ne zaman, hangi işlemi yaptı"
sorusuna kod içinden değil audit log'dan cevap verilebilmesini sağlıyor. Kayıtta
`request_id` alanı BİLEREK boş bırakılıyor (bu fonksiyon orijinal sohbet turunun
dışında, insan onayı anında çağrılıyor — o turun `request_id`'si burada bilinmiyor);
bunun yerine `approval_request_id` alanı kullanılıyor — `tool_agent.py`'nin
`submit_approval_*` kaydındaki AYNI id ile eşleştirilip iki olay (talep + karar)
zincirlenebilir.
"""

from __future__ import annotations

from app.human_in_the_loop.approval_queue import ApprovalRequest, get, update
from app.observability import audit_log
from app.tools import reservation


def _ensure_pending(request: ApprovalRequest) -> None:
    if request.status != "pending":
        raise ValueError(f"Talep zaten sonuçlandırılmış: {request.id} ({request.status})")


def approve(request_id: str) -> ApprovalRequest:
    request = get(request_id)
    if request is None:
        raise ValueError(f"Talep bulunamadı: {request_id}")
    _ensure_pending(request)

    if request.action == "cancel":
        result = reservation.cancel_reservation(request.pnr)
    elif request.action == "change_date":
        result = reservation.change_flight_date(request.pnr, request.payload["new_date"])
    else:
        raise ValueError(f"Bilinmeyen aksiyon: {request.action}")

    request.status = "approved"
    request.result = result.model_dump() if result is not None else {"found": False}
    update(request)
    audit_log.record(
        actor="human_approver",
        action=f"approve_{request.action}",
        approval_request_id=request.id,
        pnr=request.pnr,
        result=request.result,
    )
    return request


def reject(request_id: str, reason: str | None = None) -> ApprovalRequest:
    request = get(request_id)
    if request is None:
        raise ValueError(f"Talep bulunamadı: {request_id}")
    _ensure_pending(request)

    request.status = "rejected"
    request.rejection_reason = reason
    update(request)
    audit_log.record(
        actor="human_approver",
        action=f"reject_{request.action}",
        approval_request_id=request.id,
        pnr=request.pnr,
        reason=reason,
    )
    return request
