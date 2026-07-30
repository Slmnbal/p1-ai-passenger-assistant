"""LangGraph konuşma state şeması — Adım 6.

Neden TypedDict (Pydantic degil): LangGraph'in StateGraph'i her node'un dondurdugu
sozlugu bir onceki state'in uzerine shallow-merge eder; bunun icin TypedDict + gerekirse
`Annotated[..., operator.add]` reducer'i yeterli ve LangGraph ornekleriyle birebir
uyumlu. Pydantic modeli kullanmak ekstra validation katmani ekler ama bu asamada
(Adim 6) gerekli degil — girdi zaten kendi ic sistemimizden (tool/RAG/model ciktilari)
geliyor, disaridan gelen dogrulanmamis veri degil.

`history` alani `operator.add` ile isaretlendi: her node kendi turn'unu ekler
(`{"role": ..., "content": ...}` sozlugu iceren tek elemanli bir liste dondurur),
LangGraph bunu bir onceki gecmisin SONUNA ekler — node'un butun gecmisi yeniden
yazmasina gerek kalmaz.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class RetrievedSource(TypedDict):
    source_file: str
    section_title: str
    score: float
    text: str


class ConversationState(TypedDict, total=False):
    # Girdi
    user_message: str
    session_preferences: dict[str, str]
    history: Annotated[list[dict[str, str]], operator.add]

    # Adim 9: gozlemlenebilirlik icin correlation id — JSON loglar, audit log ve
    # Langfuse trace'i AYNI id ile birbirine baglar (bkz. app/observability/)
    request_id: str

    # planning_agent ciktisi
    intent: str | None
    intent_confidence: float | None
    entities: dict[str, str]

    # retrieval_agent ciktisi
    retrieved_sources: list[RetrievedSource]
    policy_answer: str | None

    # tool_agent ciktisi
    tool_result: dict | None

    # policy_verification_agent ciktisi
    grounded: bool | None
    needs_clarification: bool
    clarification_question: str | None
    retry_count: int

    # graph.py'nin son adimi
    final_response: str | None

    # guardrails katmani (Adim 7)
    blocked: bool
    block_reason: str | None
    guardrail_checks: dict[str, bool]
    low_confidence_intent: bool
