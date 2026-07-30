"""Mock tool'ların girdi/cikti semalari (Pydantic).

Neden ayri bir dosyada: Her tool modulu (flight_search, reservation, checkin) kendi
semasini kullanir; semalari tek yerde tutmak, LangGraph'in (Adim 6) tool tanimlarini
buradan okuyup import etmesini kolaylastirir. Ayrica FastAPI, bu siniflardan otomatik
olarak OpenAPI/JSON semasi uretir — plan'in "her tool icin JSON semasi" istedigi cikti
budur, elle ayrica yazilmiyor.

`requires_human_approval` alani her "islem" cevabinda var: Adim 8'de insan onay kuyrugu
bu alana bakip kritik islemleri (iptal/iade/tarih degisikligi) durdurup onaya sokacak.
Su an (Adim 5'te) bu alan sadece BILGI olarak donuyor, henuz hicbir sey durdurmuyor —
onay mekanizmasinin kendisi Adim 8'in isi.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FlightSearchRequest(BaseModel):
    origin: str = Field(..., description="Kalkış havalimanı kodu (örn. IST)")
    destination: str = Field(..., description="Varış havalimanı kodu (örn. ESB)")


class FlightInfo(BaseModel):
    flight_no: str
    origin: str
    destination: str
    typical_duration_min: int
    status: Literal["on_time", "delayed", "cancelled"]
    delay_minutes: int = 0
    gate: str | None = None


class FlightSearchResponse(BaseModel):
    query: FlightSearchRequest
    flights: list[FlightInfo]


class FlightStatusRequest(BaseModel):
    flight_no: str = Field(..., description="Uçuş numarası (örn. TK2110)")


class ReservationInfo(BaseModel):
    pnr: str
    passenger_name: str
    flight_no: str
    date: str
    cabin_class: str
    fare_type: str
    seat: str
    checked_in: bool
    baggage_pieces: int


class ReservationLookupRequest(BaseModel):
    pnr: str


class CancelReservationRequest(BaseModel):
    pnr: str


class CancelReservationResponse(BaseModel):
    pnr: str
    status: Literal["cancelled"]
    refund_amount_estimate_try: float
    requires_human_approval: bool = True
    message: str


class ChangeFlightDateRequest(BaseModel):
    pnr: str
    new_date: str = Field(..., description="YYYY-MM-DD formatında yeni tarih")


class ChangeFlightDateResponse(BaseModel):
    pnr: str
    old_date: str
    new_date: str
    requires_human_approval: bool = True
    message: str


class AddBaggageRequest(BaseModel):
    pnr: str
    extra_pieces: int = Field(..., ge=1, le=5)


class AddBaggageResponse(BaseModel):
    pnr: str
    new_baggage_pieces: int
    estimated_fee_try: float
    requires_human_approval: bool = False
    message: str


class CheckInRequest(BaseModel):
    pnr: str


class BoardingPass(BaseModel):
    pnr: str
    passenger_name: str
    flight_no: str
    seat: str
    gate: str
    boarding_time: str


class CheckInResponse(BaseModel):
    status: Literal["success", "already_checked_in", "not_yet_open", "window_closed"]
    boarding_pass: BoardingPass | None = None
    message: str


class PolicyQueryRequest(BaseModel):
    question: str


class PolicySource(BaseModel):
    source_file: str
    section_title: str
    score: float


class PolicyQueryResponse(BaseModel):
    question: str
    top_sources: list[PolicySource]
