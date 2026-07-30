"""Rezervasyon yonetimi mock araci: sorgulama, iptal, tarih degisikligi, ekstra bagaj.

`requires_human_approval` degeri, p1_proje_plani.md Adim 6/8'deki karara gore sabit
atanir (kod icinde hesaplanmiyor, cunku bu bir IS KURALI karari, veri degil):
- Iptal ve tarih degisikligi: True (plan: "kritik islemler" — geri donusu zor/mali etkisi var)
- Ekstra bagaj: False (dusuk riskli, satin alma islemi, plan bunu "kritik" saymiyor)

Bu alan Adim 5'te SADECE BILGI tasir, hicbir sey durdurmaz — onay kuyrugu Adim 8'de
eklenecek. Simdiden bu alani donduruyoruz cunku LangGraph (Adim 6), tool sonucuna
bakip "bunu onaya gondermeli miyim" kararini verecek; alan olmadan bu karar imkansiz
olurdu.
"""

from __future__ import annotations

from app.tools import store
from app.tools.schemas import (
    AddBaggageResponse,
    CancelReservationResponse,
    ChangeFlightDateResponse,
    ReservationInfo,
)

# Mock ucret tablosu: gercek THY ucretleri degil, illustratif sabit degerler.
# Gercekci bir iade hesaplamasi icin data/policies/fare_conditions_policy.md'deki
# kurallara baglanmak gerekir — bu, Adim 5'in kapsami disinda (basit mock yeterli).
_MOCK_BASE_FARE_TRY = {"economy": 3000.0, "business": 9000.0}
_MOCK_REFUND_RATE = {
    "EcoFly": 0.0, "ExtraFly": 0.5, "PrimeFly": 0.7,
    "BusinessFly": 0.8, "BusinessPrime": 0.9, "Business": 0.9,
}


def get_reservation(pnr: str) -> ReservationInfo | None:
    record = store.RESERVATIONS.get(pnr)
    if record is None:
        return None
    return ReservationInfo(**record)


def cancel_reservation(pnr: str) -> CancelReservationResponse | None:
    record = store.RESERVATIONS.get(pnr)
    if record is None:
        return None

    base_fare = _MOCK_BASE_FARE_TRY.get(record["cabin_class"], 3000.0)
    refund_rate = _MOCK_REFUND_RATE.get(record["fare_type"], 0.0)
    refund = round(base_fare * refund_rate, 2)

    del store.RESERVATIONS[pnr]

    return CancelReservationResponse(
        pnr=pnr,
        status="cancelled",
        refund_amount_estimate_try=refund,
        message=(
            f"Rezervasyon iptal edildi. Tahmini iade tutarı {refund} TRY "
            f"({record['fare_type']} ücret sınıfı kuralına göre) — insan onayı bekliyor."
        ),
    )


def change_flight_date(pnr: str, new_date: str) -> ChangeFlightDateResponse | None:
    record = store.RESERVATIONS.get(pnr)
    if record is None:
        return None

    old_date = record["date"]
    record["date"] = new_date

    return ChangeFlightDateResponse(
        pnr=pnr,
        old_date=old_date,
        new_date=new_date,
        message=f"Uçuş tarihi {old_date} → {new_date} olarak güncellendi — insan onayı bekliyor.",
    )


def add_baggage(pnr: str, extra_pieces: int) -> AddBaggageResponse | None:
    record = store.RESERVATIONS.get(pnr)
    if record is None:
        return None

    fee_per_piece = 90.0  # mock ABD/dis hat ekstra bagaj ucreti (bkz. baggage_policy.md, illustratif)
    record["baggage_pieces"] += extra_pieces
    fee = round(fee_per_piece * extra_pieces, 2)

    return AddBaggageResponse(
        pnr=pnr,
        new_baggage_pieces=record["baggage_pieces"],
        estimated_fee_try=fee,
        message=f"{extra_pieces} parça ekstra bagaj eklendi, tahmini ücret {fee} TRY.",
    )
