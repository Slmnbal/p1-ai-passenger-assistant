"""P1 FastAPI servisi.

Tasarim karari: tool fonksiyonlari (app/tools/*.py) PNR/ucus bulunamadiginca `None`
dondurur, HTTPException FIRLATMAZ. Neden: o fonksiyonlar HTTP'den tamamen habersiz
olmali — LangGraph (Adim 6) bu fonksiyonlari HTTP uzerinden degil, dogrudan Python
cagrisi olarak kullaniyor; o zaman "404" gibi bir HTTP kavraminin bir anlami yok. Bu
yuzden None->404 cevirisi SADECE burada, main.py'da yapilir — is mantigi ile HTTP
katmani birbirinden ayri tutulur (separation of concerns).

**Adim 10:** `/chat` endpoint'i tum agent katmanini (planning/RAG/tool/guardrail,
Adim 6-7) tek bir HTTP cagrisinda birlestiriyor — `app/agent/graph.py`'nin `run()`'ini
dogrudan cagirir, hicbir ek is mantigi burada YOK (o mantik zaten test edilmis
node'larda). `/approvals/*` endpoint'leri Adim 8'in onay kuyruğunu HTTP uzerinden
kullanilabilir hale getiriyor — plan bunu acikca istemedi ama insan-onay akisinin
sadece Python fonksiyonu olarak degil, gercek bir servis olarak da calisir olmasi
icin eklendi.

**Adim 12:** `app/static/index.html` — dahili kullanim icin basit bir web konsolu
(`/ui`). Ayri bir frontend build sureci (React vb.) BILINCLI olarak kurulmadi: proje
tek bir HTML dosyasinda gomulu CSS/JS ile calisiyor, cunku ihtiyac sadece `/chat` ve
`/approvals/*` endpoint'lerini gorsel olarak kullanabilmek — ekstra bir build araci
bu olcekte gereksiz karmasiklik olurdu.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from app.agent.graph import run as run_agent_graph
from app.human_in_the_loop import approval_flow, approval_queue
from app.observability import metrics
from app.tools import checkin, flight_search, policy_lookup, reservation
from app.tools.schemas import (
    AddBaggageRequest,
    AddBaggageResponse,
    CancelReservationResponse,
    ChangeFlightDateRequest,
    ChangeFlightDateResponse,
    CheckInResponse,
    FlightInfo,
    FlightSearchRequest,
    FlightSearchResponse,
    PolicyQueryRequest,
    PolicyQueryResponse,
    ReservationInfo,
)

app = FastAPI(
    title="P1 — AI Passenger Experience Assistant",
    description="Bağımsız portföy projesi; Turkish Airlines ile resmi bağlantısı yoktur.",
    version="1.0.0",
)

app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Kullanıcının yazdığı mesaj")
    session_id: str | None = Field(
        None, description="Önceki turu hatırlamak için — ilk mesajda boş bırakılır, "
        "sunucunun döndürdüğü session_id sonraki isteklerde geri gönderilir"
    )


class ChatSourceRef(BaseModel):
    source_file: str
    section_title: str
    score: float


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    message: str
    intent: str | None = None
    blocked: bool = False
    block_reason: str | None = None
    needs_clarification: bool = False
    sources: list[ChatSourceRef] = []


class ApprovalActionRequest(BaseModel):
    reason: str | None = None


@app.post("/tools/flights/search", response_model=FlightSearchResponse)
def search_flights(request: FlightSearchRequest) -> FlightSearchResponse:
    return flight_search.search_flights(request)


@app.get("/tools/flights/{flight_no}/status", response_model=FlightInfo)
def flight_status(flight_no: str) -> FlightInfo:
    result = flight_search.get_flight_status(flight_no)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Uçuş bulunamadı: {flight_no}")
    return result


@app.get("/tools/reservations/{pnr}", response_model=ReservationInfo)
def get_reservation(pnr: str) -> ReservationInfo:
    result = reservation.get_reservation(pnr)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Rezervasyon bulunamadı: {pnr}")
    return result


@app.post("/tools/reservations/{pnr}/cancel", response_model=CancelReservationResponse)
def cancel_reservation(pnr: str) -> CancelReservationResponse:
    result = reservation.cancel_reservation(pnr)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Rezervasyon bulunamadı: {pnr}")
    return result


@app.post("/tools/reservations/{pnr}/change-date", response_model=ChangeFlightDateResponse)
def change_flight_date(pnr: str, request: ChangeFlightDateRequest) -> ChangeFlightDateResponse:
    result = reservation.change_flight_date(pnr, request.new_date)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Rezervasyon bulunamadı: {pnr}")
    return result


@app.post("/tools/reservations/{pnr}/add-baggage", response_model=AddBaggageResponse)
def add_baggage(pnr: str, request: AddBaggageRequest) -> AddBaggageResponse:
    result = reservation.add_baggage(pnr, request.extra_pieces)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Rezervasyon bulunamadı: {pnr}")
    return result


@app.post("/tools/checkin/{pnr}", response_model=CheckInResponse)
def check_in(pnr: str) -> CheckInResponse:
    result = checkin.check_in(pnr)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Rezervasyon bulunamadı: {pnr}")
    return result


@app.post("/tools/policy/query", response_model=PolicyQueryResponse)
def query_policy(request: PolicyQueryRequest) -> PolicyQueryResponse:
    return policy_lookup.query_policy(request)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    final_state = run_agent_graph(request.message, session_id=request.session_id)

    sources = []
    if final_state.get("grounded"):
        sources = [
            ChatSourceRef(
                source_file=s["source_file"], section_title=s["section_title"], score=s["score"]
            )
            for s in final_state.get("retrieved_sources") or []
        ]

    return ChatResponse(
        request_id=final_state["request_id"],
        session_id=final_state["session_id"],
        message=final_state.get("final_response") or "",
        intent=final_state.get("intent"),
        blocked=final_state.get("blocked", False),
        block_reason=final_state.get("block_reason"),
        needs_clarification=final_state.get("needs_clarification", False),
        sources=sources,
    )


@app.get("/approvals/pending", response_model=list[approval_queue.ApprovalRequest])
def list_pending_approvals() -> list[approval_queue.ApprovalRequest]:
    return approval_queue.list_pending()


def _approval_error_status(message: str) -> int:
    return 409 if "sonuçlandırılmış" in message else 404


@app.post("/approvals/{request_id}/approve", response_model=approval_queue.ApprovalRequest)
def approve_request(request_id: str) -> approval_queue.ApprovalRequest:
    try:
        return approval_flow.approve(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=_approval_error_status(str(exc)), detail=str(exc)) from exc


@app.post("/approvals/{request_id}/reject", response_model=approval_queue.ApprovalRequest)
def reject_request(request_id: str, request: ApprovalActionRequest | None = None) -> approval_queue.ApprovalRequest:
    try:
        return approval_flow.reject(request_id, reason=request.reason if request else None)
    except ValueError as exc:
        raise HTTPException(status_code=_approval_error_status(str(exc)), detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics() -> Response:
    """Adım 9: Prometheus'un scrape ettiği endpoint (bkz. docker/prometheus/prometheus.yml)."""
    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)
