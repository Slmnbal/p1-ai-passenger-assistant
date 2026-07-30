"""Audit log — kim, ne zaman, hangi işlemi yaptı (Adım 9).

JSONL (bir satır = bir olay) formatı seçildi çünkü `app/tools/store.py`'nin JSON dosya
deseniyle tutarlı ve append-only bir log için (rastgele erişim gerekmiyor, sadece sırayla
ekleme) en basit çözüm. Gerçek bir üretimde bu bir veritabanı tablosu olurdu — mock
projede dosya yeterli (bkz. `app/tools/store.py`'nin de "gerçek DB Adım 10+'ın işi"
gerekçesi).

`actor` alanı ÖNEMLİ: `human_approver` (Adım 8'deki approve/reject — gerçek bir insan
kararı) ile `system` (kullanıcı mesajından otomatik tetiklenen tool çağrısı) ayrımını
nettir — ilerideki bir güvenlik incelemesinde "bunu kim yaptı" sorusuna kod içinden değil
audit log'dan cevap verilebilmesi gerekiyor.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_AUDIT_LOG_PATH = Path("logs/audit_log.jsonl")


def record(actor: str, action: str, request_id: str | None = None, **details) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "action": action,
        "request_id": request_id,
        **details,
    }
    _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_all() -> list[dict]:
    if not _AUDIT_LOG_PATH.exists():
        return []
    with open(_AUDIT_LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reset() -> None:
    """Testler için: log dosyasını temizler (bkz. app/tools/store.py::reset ile aynı gerekçe)."""
    if _AUDIT_LOG_PATH.exists():
        _AUDIT_LOG_PATH.unlink()
