"""Adım 6 LangGraph agent katmanı icin testler.

planning/tool testleri tamamen deterministik ve yerel (fine-tuned model diskten
yukleniyor, mock store bellek ici) — dis servise ihtiyac yok. `Live*` ile baslayan
sinif altindaki testler ise GERCEK Qdrant (Docker) ve Ollama'ya baglanir — bkz. proje
hafizasi "professional-rigor-mentality": RAG'i mock'lamak, tam da olcmeye calistigimiz
kaynak-atif/groundedness davranisini gizlerdi. Adim 10'da CI'a baglaninca bu servisler
container olarak ayaga kaldirilacak; simdilik (Adim 6) yerel makinede Qdrant+Ollama
acikken calistirilir.
"""

from __future__ import annotations

import pytest

from app.agent import tool_agent
from app.agent.planning_agent import classify_intent, plan_node, route_after_planning
from app.agent.policy_verification_agent import verify_node
from app.human_in_the_loop import approval_queue
from app.tools import store


@pytest.fixture(autouse=True)
def reset_store():
    store.reset()
    approval_queue.reset()
    yield


# --- planning_agent (gercek fine-tuned model — dis SERVIS yok ama `models/intent_full_10ep/`
# agirliklari (422MB) .gitignore'da, CI'da yok; bu yuzden `live` ile isaretli, bkz. Adim 10) ---

@pytest.mark.live
@pytest.mark.parametrize(
    "text,expected_intent",
    [
        ("TK2110 rötarlı mı?", "ucus_sorgulama"),
        ("SYN1A2B için check-in yapmak istiyorum", "checkin_talebi"),
        ("SYN3C4D rezervasyonumu iptal etmek istiyorum", "rezervasyon_islem_talebi"),
        ("Business class bagaj hakkım nedir?", "politika_bilgi_sorgusu"),
        ("Bir sorum var", "belirsiz_acikliga_kavusturma"),
        ("THY kargo işine nasıl başvurabilirim?", "kapsam_disi"),
    ],
)
def test_classify_intent_covers_all_classes(text, expected_intent):
    intent, confidence = classify_intent(text)
    assert intent == expected_intent
    assert 0.0 < confidence <= 1.0


@pytest.mark.live
def test_plan_node_kapsam_disi_sets_final_response_directly():
    result = plan_node({"user_message": "Turkish Airlines hisse senedi bugün ne kadar işlem görüyor?"})
    assert result["intent"] == "kapsam_disi"
    assert result["final_response"] is not None


@pytest.mark.live
def test_plan_node_belirsiz_sets_clarification():
    result = plan_node({"user_message": "yardım lazım"})
    assert result["intent"] == "belirsiz_acikliga_kavusturma"
    assert result["needs_clarification"] is True
    assert result["clarification_question"] == result["final_response"]


@pytest.mark.parametrize(
    "intent,expected_route",
    [
        ("kapsam_disi", "end"),
        ("belirsiz_acikliga_kavusturma", "end"),
        ("politika_bilgi_sorgusu", "retrieval"),
        ("ucus_sorgulama", "tools"),
        ("rezervasyon_islem_talebi", "tools"),
        ("checkin_talebi", "tools"),
    ],
)
def test_route_after_planning(intent, expected_route):
    assert route_after_planning({"intent": intent}) == expected_route


# --- tool_agent (mock store, dis servis yok) ---

def test_extract_pnr_and_flight_no_and_date():
    assert tool_agent._extract_pnr("SYN1A2B için...") == "SYN1A2B"
    assert tool_agent._extract_flight_no("TK2110 rötarlı mı?") == "TK2110"
    assert tool_agent._extract_flight_no("THY 2110 rötarlı mı?") == "TK2110"
    assert tool_agent._extract_date("2026-09-01 tarihine almak istiyorum") == "2026-09-01"


def test_tool_node_flight_status_found():
    out = tool_agent.tool_node({"user_message": "TK2110 rötarlı mı?", "intent": "ucus_sorgulama"})
    assert out["tool_result"]["flight_info"]["flight_no"] == "TK2110"
    assert "final_response" in out


def test_tool_node_flight_status_not_found():
    out = tool_agent.tool_node({"user_message": "TK9999 rötarlı mı?", "intent": "ucus_sorgulama"})
    assert out["tool_result"] == {"found": False}


def test_tool_node_flight_search_by_route():
    out = tool_agent.tool_node({"user_message": "IST ESB arası uçuş var mı?", "intent": "ucus_sorgulama"})
    assert len(out["tool_result"]["flights"]) == 1
    assert out["tool_result"]["flights"][0]["flight_no"] == "TK2110"


def test_tool_node_flight_query_without_entities_asks_clarification():
    out = tool_agent.tool_node({"user_message": "uçuşum ne zaman kalkıyor", "intent": "ucus_sorgulama"})
    assert out["needs_clarification"] is True


def test_tool_node_checkin_missing_pnr_asks_clarification():
    out = tool_agent.tool_node({"user_message": "check-in yapmak istiyorum", "intent": "checkin_talebi"})
    assert out["needs_clarification"] is True


def test_tool_node_checkin_success_after_flight_date_reached():
    out = tool_agent.tool_node(
        {"user_message": "SYN5E6F için check-in yapmak istiyorum", "intent": "checkin_talebi"}
    )
    assert out["tool_result"]["checkin"]["status"] in ("success", "already_checked_in")


def test_tool_node_reservation_missing_pnr_asks_clarification():
    out = tool_agent.tool_node(
        {"user_message": "rezervasyonumu iptal etmek istiyorum", "intent": "rezervasyon_islem_talebi"}
    )
    assert out["needs_clarification"] is True


def test_tool_node_reservation_cancel_submits_to_approval_queue_without_executing():
    """Adım 8: iptal artık HEMEN çalışmıyor, onay kuyruğuna giriyor — store değişmez."""
    out = tool_agent.tool_node(
        {"user_message": "SYN3C4D rezervasyonumu iptal etmek istiyorum", "intent": "rezervasyon_islem_talebi"}
    )
    request = out["tool_result"]["approval_request"]
    assert request["status"] == "pending"
    assert request["action"] == "cancel"
    assert store.RESERVATIONS.get("SYN3C4D") is not None  # henüz iptal EDİLMEDİ


def test_tool_node_reservation_change_date_missing_date_asks_clarification():
    out = tool_agent.tool_node(
        {"user_message": "SYN5E6F için tarihi değiştirmek istiyorum", "intent": "rezervasyon_islem_talebi"}
    )
    assert out["needs_clarification"] is True


def test_tool_node_reservation_change_date_submits_to_approval_queue_without_executing():
    """Adım 8: tarih değişikliği de HEMEN çalışmıyor — mevcut tarih değişmeden kalır."""
    out = tool_agent.tool_node(
        {
            "user_message": "SYN5E6F için tarihi 2026-09-01 olarak değiştirmek istiyorum",
            "intent": "rezervasyon_islem_talebi",
        }
    )
    request = out["tool_result"]["approval_request"]
    assert request["status"] == "pending"
    assert request["payload"]["new_date"] == "2026-09-01"
    assert store.RESERVATIONS["SYN5E6F"]["date"] != "2026-09-01"


def test_tool_node_reservation_add_baggage():
    out = tool_agent.tool_node(
        {
            "user_message": "SYN1A2B için 2 parça ekstra bagaj eklemek istiyorum",
            "intent": "rezervasyon_islem_talebi",
        }
    )
    assert out["tool_result"]["add_baggage"]["new_baggage_pieces"] == 3  # 1 (mevcut) + 2


def test_tool_node_reservation_unknown_action_asks_clarification():
    out = tool_agent.tool_node(
        {"user_message": "SYN1A2B rezervasyonumda bir şey yapmak istiyorum", "intent": "rezervasyon_islem_talebi"}
    )
    assert out["needs_clarification"] is True


# --- policy_verification_agent (sentetik sources, dis servis yok) ---

def test_verify_node_detects_llm_fallback_phrase():
    state = {
        "policy_answer": "Bu konuda kaynaklarımda net bir bilgi bulamadım.",
        "retrieved_sources": [{"source_file": "x.md", "section_title": "y", "score": 0.4, "text": "alakasız metin"}],
        "retry_count": 0,
    }
    out = verify_node(state)
    assert out["grounded"] is False
    assert out["final_response"] == state["policy_answer"]


def test_verify_node_grounded_answer_passes():
    source_text = "Business class kabin bagajı 8 kg ve 23x40x55 cm ölçülerindedir."
    state = {
        "policy_answer": "Business class kabin bagajı 8 kg ve 23x40x55 cm ölçüsündedir.",
        "retrieved_sources": [{"source_file": "x.md", "section_title": "y", "score": 0.7, "text": source_text}],
        "retry_count": 0,
    }
    out = verify_node(state)
    assert out["grounded"] is True
    assert "Kaynak:" in out["final_response"]


def test_verify_node_ungrounded_answer_triggers_retry_then_fallback():
    state = {
        "policy_answer": "Uçakta ücretsiz kablosuz internet ve sınırsız atıştırmalık sunulur.",
        "retrieved_sources": [{"source_file": "x.md", "section_title": "y", "score": 0.5, "text": "bagaj kuralları hakkında tamamen alakasız bir metin"}],
        "retry_count": 0,
    }
    first = verify_node(state)
    assert first["grounded"] is False
    assert first.get("final_response") is None
    assert first["retry_count"] == 1

    state.update(first)
    second = verify_node(state)
    assert second["grounded"] is False
    assert second["final_response"] == "Bu konuda kaynaklarımda net bir bilgi bulamadım."


def test_verify_node_conflicting_sources_ask_clarification():
    state = {
        "policy_answer": "Economy sınıfında 1 parça, Business sınıfında 2 parça bagaj hakkınız var.",
        "retrieved_sources": [
            {"source_file": "a.md", "section_title": "Economy bagaj hakkı", "score": 0.601, "text": "economy 1 parça"},
            {"source_file": "b.md", "section_title": "Business bagaj hakkı", "score": 0.599, "text": "business 2 parça"},
        ],
        "retry_count": 0,
    }
    out = verify_node(state)
    assert out["needs_clarification"] is True


@pytest.mark.live
class TestLiveRagPipeline:
    """Bu sinifin testleri GERCEK Qdrant + Ollama'ya baglanir (yerelde acik olmali)."""

    def test_retrieve_and_answer_grounds_in_real_corpus(self):
        from app.agent.retrieval_agent import retrieve_and_answer

        out = retrieve_and_answer({"user_message": "Business class kabin bagajı kaç kg olabilir?"})
        assert len(out["retrieved_sources"]) == 3
        assert "8" in out["policy_answer"]

    def test_graph_end_to_end_policy_question(self):
        from app.agent.graph import run

        final = run("Business class bagaj hakkım nedir?")
        assert final["intent"] == "politika_bilgi_sorgusu"
        assert final["grounded"] is True
        assert "Kaynak:" in final["final_response"]

    def test_graph_end_to_end_tool_question_does_not_need_live_rag(self):
        from app.agent.graph import run

        final = run("TK2110 rötarlı mı?")
        assert final["intent"] == "ucus_sorgulama"
        assert final["final_response"] is not None
