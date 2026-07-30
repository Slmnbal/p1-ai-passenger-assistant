"""Onay kuyruğu — kritik işlemler (iptal, tarih değişikliği) artık HEMEN uygulanmıyor,
burada bekletiliyor. `app/tools/store.py`'deki "in-memory, tek kaynak" desenini izler.

Adım 5'te `requires_human_approval` alanı yalnızca BİLGİ taşıyordu (bkz.
`reservation.py` docstring'i: "onay kuyruğu Adım 8'de eklenecek"). Adım 6'da
`tool_agent.py` bu alana bakmadan cancel/change_date'i doğrudan çalıştırıyordu. Bu
modülle birlikte gerçek mutasyon artık SADECE `approval_flow.approve()` çağrıldığında
gerçekleşiyor — `submit()` yalnızca bir talep oluşturur, store'da hiçbir şeyi değiştirmez.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from app.observability.metrics import APPROVAL_QUEUE_PENDING


class ApprovalRequest(BaseModel):
    id: str
    action: Literal["cancel", "change_date"]
    pnr: str
    payload: dict = {}
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: str
    result: dict | None = None
    rejection_reason: str | None = None


_QUEUE: dict[str, ApprovalRequest] = {}


def _sync_gauge() -> None:
    """Adım 9: Prometheus gauge'unu her değişiklikten sonra GERÇEK sayıdan yeniden
    hesaplar (artır/azalt yerine) — sürüklenme (drift) riskini ortadan kaldırır."""
    APPROVAL_QUEUE_PENDING.set(len(list_pending()))


def submit(action: str, pnr: str, payload: dict | None = None) -> ApprovalRequest:
    request = ApprovalRequest(
        id=str(uuid.uuid4())[:8],
        action=action,
        pnr=pnr,
        payload=payload or {},
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _QUEUE[request.id] = request
    _sync_gauge()
    return request


def list_pending() -> list[ApprovalRequest]:
    return [r for r in _QUEUE.values() if r.status == "pending"]


def get(request_id: str) -> ApprovalRequest | None:
    return _QUEUE.get(request_id)


def update(request: ApprovalRequest) -> None:
    _QUEUE[request.id] = request
    _sync_gauge()


def reset() -> None:
    """Testler için: kuyruğu sıfırlar (bkz. app/tools/store.py::reset ile aynı gerekçe)."""
    _QUEUE.clear()
    _sync_gauge()
