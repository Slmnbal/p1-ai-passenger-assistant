"""LangGraph StateGraph wiring — Adım 6'nın omurgası, Adım 7'nin guardrail node'ları ve
Adım 9'un gözlemlenebilirlik (JSON log + Langfuse trace + Prometheus metrik) katmanıyla
genişletildi.

Akış:

    input_guard ─ injection tespit edildi ────────────────────────────────→ BİTTİ
               └─ planning ─┬─ kapsam_disi ───────────────────────────────→ BİTTİ
                             ├─ politika_bilgi_sorgusu / belirsiz_acikliga_kavusturma
                             │     → retrieval → verification ─┬─ retry (max 1) ┘
                             │                                 └─ output_guard → BİTTİ
                             └─ ucus/rezervasyon/checkin → tools → output_guard ──→ BİTTİ

`retrieval` node'u, `retry_count`'a göre top_k'yı artırarak çağrılır (3 → 5) — reflection
loop'un anlamlı olması için (bkz. policy_verification_agent docstring'i).

`output_guard`, yalnızca GERÇEKTEN bir tool işlemi ya da kaynağa dayalı bir politika
cevabı üretildiğinde çalışır (`kapsam_disi`'nin sabit, LLM'siz cevabı zaten risksiz
olduğu için bu node'a hiç uğramaz — bkz. `route_after_planning`).

**1 Ağustos 2026 (bkz. docs/adr/0006-...md):** `belirsiz_acikliga_kavusturma` da artık
`retrieval`'a uğruyor — intent sınıflandırıcının bariz politika sorularını bile bu
sınıfa düşürebildiği canlı testte gözlemlendi (bkz. MODEL_CARD.md'deki ~%24 hata
oranı). RAG grounded bir cevap bulursa onu döndürür, bulamazsa `verify_node` mevcut
netleştirme mesajına düşer — bu bir güvenlik ağı, RAG'in kendisi değiştirilmedi.

**Adım 9:** `run()`, her çağrıda bir `request_id` (uuid) üretir ve bunu bir Langfuse
trace'ine bağlar. `_instrument()`, her node'u şeffaf bir şekilde sarar: node çalışma
süresini Prometheus histogramına yazar, JSON log satırı üretir ve Langfuse'a bir span
gönderir — tek tek her node fonksiyonunu (Adım 6-7'de zaten test edilmiş) değiştirmeden.
"""

from __future__ import annotations

import logging
import uuid

from langgraph.graph import END, StateGraph

from app.agent import session_memory
from app.agent.planning_agent import plan_node, route_after_planning
from app.agent.policy_verification_agent import route_after_verification, verify_node
from app.agent.retrieval_agent import retrieve_and_answer
from app.agent.state import ConversationState
from app.agent.tool_agent import tool_node
from app.guardrails.confidence_fallback import flag_low_confidence_intent
from app.guardrails.grounding import judge_faithfulness, require_sources
from app.guardrails.numeric_template import check_numeric_consistency
from app.guardrails.prompt_injection_tests import detect_injection
from app.guardrails.schemas import check_business_rules
from app.observability import metrics, tracing
from app.observability.structured_logging import configure_logging, get_logger, log_event

configure_logging()
_logger = get_logger("graph")

_BLOCKED_MESSAGE = (
    "Bu isteği güvenlik nedeniyle işleyemiyorum. Lütfen sorunuzu normal bir yolcu "
    "hizmeti talebi olarak yeniden iletir misiniz?"
)
_UNVERIFIABLE_MESSAGE = "Bu konuda kaynaklarımda net bir bilgi bulamadım."


def input_guard_node(state: ConversationState) -> dict:
    flagged, category = detect_injection(state["user_message"])
    if flagged:
        return {
            "blocked": True,
            "block_reason": f"prompt_injection:{category}",
            "guardrail_checks": {"injection_detected": True},
            "final_response": _BLOCKED_MESSAGE,
        }
    return {"blocked": False, "guardrail_checks": {"injection_detected": False}}


def route_after_input_guard(state: ConversationState) -> str:
    return "end" if state.get("blocked") else "planning"


def _retrieval_node(state: ConversationState) -> dict:
    top_k = 3 + 2 * state.get("retry_count", 0)
    return retrieve_and_answer(state, top_k=top_k)


def output_guard_node(state: ConversationState) -> dict:
    blocking_checks: dict[str, bool] = {}
    observed_checks: dict[str, bool] = {}
    reasons: list[str] = []
    request_id = state.get("request_id")

    tool_result = state.get("tool_result")
    if tool_result:
        report = check_business_rules(state.get("intent"), tool_result)
        blocking_checks.update(report.checks)
        if not report.passed:
            reasons.append(report.reason)

    if state.get("grounded") is True and state.get("policy_answer") is not None:
        sources = state.get("retrieved_sources", [])
        blocking_checks["has_sources"] = require_sources(sources)
        if not blocking_checks["has_sources"]:
            reasons.append("kaynak yok")

        numeric_ok, unsupported = check_numeric_consistency(state["policy_answer"], sources)
        blocking_checks["numeric_consistency"] = numeric_ok
        if not numeric_ok:
            reasons.append(f"desteklenmeyen sayılar: {', '.join(unsupported)}")

        # judge_faithfulness artık BLOKLAYICI — claim decomposition'la %90 doğruluğa
        # çıkarıldıktan sonra (bkz. grounding.py, docs/adr/0003 Deney 4) gözlem-only
        # statüsünden bloklayıcı statüye terfi etti.
        try:
            faithful, claim_results = judge_faithfulness(
                state["policy_answer"], sources, request_id=request_id
            )
            blocking_checks["faithfulness_judge"] = faithful
            if not faithful:
                unsupported_claims = [c["claim"] for c in claim_results if not c["supported"]]
                reasons.append(f"desteklenmeyen iddialar: {'; '.join(unsupported_claims)}")
        except Exception:
            pass

    if state.get("intent_confidence") is not None:
        observed_checks["low_confidence_intent"] = flag_low_confidence_intent(state["intent_confidence"])

    all_checks = {**blocking_checks, **{f"observed_{k}": v for k, v in observed_checks.items()}}
    passed = all(blocking_checks.values()) if blocking_checks else True

    if passed:
        return {"blocked": False, "guardrail_checks": all_checks}

    return {
        "blocked": True,
        "block_reason": "; ".join(reasons),
        "guardrail_checks": all_checks,
        "final_response": _UNVERIFIABLE_MESSAGE,
    }


def _instrument(name: str, fn):
    """Adım 9: her node'u JSON log + Prometheus latency + Langfuse span ile sarar.

    Node fonksiyonlarının kendisi (Adım 6-7'de zaten test edilmiş) HİÇ değişmiyor —
    bu, gözlemlenebilirliği "çapraz kesen" (cross-cutting) bir katman olarak, iş
    mantığının içine karıştırmadan ekliyor.
    """

    def wrapper(state: ConversationState) -> dict:
        request_id = state.get("request_id", "unknown")
        start_time = tracing.now()
        log_event(_logger, logging.INFO, f"{name} başladı", request_id=request_id)

        result = fn(state)

        end_time = tracing.now()
        duration = (end_time - start_time).total_seconds()
        metrics.NODE_LATENCY_SECONDS.labels(node=name).observe(duration)

        if name == "planning" and result.get("intent"):
            metrics.REQUESTS_TOTAL.labels(intent=result["intent"]).inc()

        if result.get("blocked"):
            metrics.GUARDRAIL_BLOCKS_TOTAL.labels(reason=result.get("block_reason", "unknown")).inc()

        log_event(
            _logger,
            logging.INFO,
            f"{name} bitti",
            request_id=request_id,
            duration_seconds=round(duration, 4),
            blocked=result.get("blocked", False),
        )
        tracing.log_span(
            request_id,
            name=f"node.{name}",
            start_time=start_time,
            end_time=end_time,
            input_data=state.get("user_message"),
            output_data={k: v for k, v in result.items() if k != "history"},
        )
        return result

    return wrapper


def build_graph():
    graph = StateGraph(ConversationState)

    graph.add_node("input_guard", _instrument("input_guard", input_guard_node))
    graph.add_node("planning", _instrument("planning", plan_node))
    graph.add_node("retrieval", _instrument("retrieval", _retrieval_node))
    graph.add_node("verification", _instrument("verification", verify_node))
    graph.add_node("tools", _instrument("tools", tool_node))
    graph.add_node("output_guard", _instrument("output_guard", output_guard_node))

    graph.set_entry_point("input_guard")

    graph.add_conditional_edges(
        "input_guard", route_after_input_guard, {"planning": "planning", "end": END}
    )
    graph.add_conditional_edges(
        "planning",
        route_after_planning,
        {"retrieval": "retrieval", "tools": "tools", "end": END},
    )
    graph.add_edge("retrieval", "verification")
    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {"retry": "retrieval", "end": "output_guard"},
    )
    graph.add_edge("tools", "output_guard")
    graph.add_edge("output_guard", END)

    return graph.compile()


def run(user_message: str, session_id: str | None = None) -> ConversationState:
    """Tek bir kullanıcı mesajını uçtan uca işler, son state'i döndürür.

    `session_id` verilirse önceki turlar (bkz. `session_memory`) `history`'ye tohum
    olarak eklenir ve `retrieval_agent` bunu cevap üretirken görebilir — takip
    sorularının bağlamı çözmesi için (bkz. `session_memory` docstring'indeki kapsam
    sınırı). `session_id` verilmezse yeni bir oturum başlatılır.
    """
    request_id = str(uuid.uuid4())
    if session_id is None:
        session_id = session_memory.new_session_id()
    prior_history = session_memory.get_history(session_id)

    tracing.start_trace(request_id, user_message)
    log_event(_logger, logging.INFO, "istek alındı", request_id=request_id, user_message=user_message)

    app = build_graph()
    final_state = app.invoke({
        "user_message": user_message,
        "retry_count": 0,
        "request_id": request_id,
        "history": prior_history,
    })

    session_memory.append_turn(session_id, user_message, final_state.get("final_response") or "")
    final_state["session_id"] = session_id

    tracing.flush()
    return final_state
