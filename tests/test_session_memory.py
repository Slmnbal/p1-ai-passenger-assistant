"""Adım 12 — çok turlu konuşma hafızası (`app/agent/session_memory.py`) birim testleri.

Gerçek LLM/Qdrant gerektirmiyor — saf in-memory store mantığı, `not live`.
"""

from __future__ import annotations

from app.agent import session_memory


def test_new_session_returns_empty_history():
    session_id = session_memory.new_session_id()
    assert session_memory.get_history(session_id) == []


def test_append_turn_stores_user_and_assistant_messages():
    session_id = session_memory.new_session_id()
    session_memory.append_turn(session_id, "Bagaj hakkım ne kadar?", "23 kg.")
    history = session_memory.get_history(session_id)
    assert history == [
        {"role": "user", "content": "Bagaj hakkım ne kadar?"},
        {"role": "assistant", "content": "23 kg."},
    ]


def test_different_sessions_do_not_share_history():
    session_a = session_memory.new_session_id()
    session_b = session_memory.new_session_id()
    session_memory.append_turn(session_a, "Soru A", "Cevap A")
    assert session_memory.get_history(session_b) == []


def test_history_capped_at_max_turns():
    session_id = session_memory.new_session_id()
    for i in range(10):
        session_memory.append_turn(session_id, f"soru {i}", f"cevap {i}")
    history = session_memory.get_history(session_id)
    assert len(history) == session_memory._MAX_TURNS * 2
    assert history[-2] == {"role": "user", "content": "soru 9"}
    assert history[-1] == {"role": "assistant", "content": "cevap 9"}
