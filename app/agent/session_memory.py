"""Çok turlu konuşma hafızası — Adım 12 Yapılacaklar #2 (ADR-0004).

`/chat` eskiden tamamen stateless'ti — her istek yeni bir `graph.run()` çağrısıydı,
önceki turu hiç bilmiyordu. Bu modül, `app/tools/store.py` ve
`app/human_in_the_loop/approval_queue.py`'deki "in-memory, tek kaynak" desenini izler:
`session_id` -> önceki turların (user/assistant) listesi.

**Bilinçli kapsam sınırı:** intent sınıflandırma (`planning_agent`) ve tool entity
çıkarımı (`tool_agent`, örn. PNR/uçuş no regex'i) hâlâ SADECE mevcut mesaja bakıyor,
geçmişi kullanmıyor — ikisi de bu bağlamda hiç eğitilmedi/tasarlanmadı, burada
genişletmek yeni, ölçülmemiş bir davranış değişikliği olurdu. Bu turda yalnızca RAG
cevap üretimi (`retrieval_agent`) önceki turları görüyor; takip soruları en iyi
ihtimalle yeni mesajın kendi başına retrieval için yeterli anahtar kelime içerdiği
durumlarda ("Ekonomi sınıfında bagaj hakkım nedir?" / "Peki business'ta?" gibi) düzgün
çalışır — tam bir sorgu yeniden yazma (query rewriting) burada YOK.
"""

from __future__ import annotations

import uuid

_SESSIONS: dict[str, list[dict[str, str]]] = {}

_MAX_TURNS = 6  # son N (user+assistant) çift — prompt boyutunu sınırlı tutmak için


def new_session_id() -> str:
    return str(uuid.uuid4())[:8]


def get_history(session_id: str) -> list[dict[str, str]]:
    return list(_SESSIONS.get(session_id, []))


def append_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    history = _SESSIONS.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})
    if len(history) > _MAX_TURNS * 2:
        del history[: len(history) - _MAX_TURNS * 2]
