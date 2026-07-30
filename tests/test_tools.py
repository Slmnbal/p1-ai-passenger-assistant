"""Adim 5 mock tool'lari icin testler.

`reset_store` fixture'i her testten once calisir (autouse=True) — reservation/checkin
testleri paylasilan store'u degistirir (iptal, check-in kalicidir), fixture olmadan
testlerin CALISMA SIRASI sonucu etkilerdi (orn. once "iptal" testi calisirsa, sonraki
test ayni PNR'i bulamaz). Bu, testlerin birbirinden bagimsiz (izole) olmasini saglar.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tools import checkin, flight_search, reservation, store
from app.tools.schemas import FlightSearchRequest


@pytest.fixture(autouse=True)
def reset_store():
    store.reset()
    yield


# --- flight_search ---

def test_search_flights_returns_matches():
    response = flight_search.search_flights(FlightSearchRequest(origin="IST", destination="ESB"))
    assert len(response.flights) == 1
    assert response.flights[0].flight_no == "TK2110"


def test_search_flights_no_match_returns_empty_list():
    response = flight_search.search_flights(FlightSearchRequest(origin="AYT", destination="JFK"))
    assert response.flights == []


def test_flight_status_unknown_flight_returns_none():
    assert flight_search.get_flight_status("TK9999") is None


def test_flight_status_is_deterministic():
    first = flight_search.get_flight_status("TK2110")
    second = flight_search.get_flight_status("TK2110")
    assert first.status == second.status
    assert first.gate == second.gate


# --- reservation ---

def test_get_reservation_found():
    r = reservation.get_reservation("SYN1A2B")
    assert r.passenger_name == "Ayşe Yılmaz"


def test_get_reservation_not_found():
    assert reservation.get_reservation("YOKPNR") is None


def test_cancel_reservation_removes_from_store_and_flags_human_approval():
    result = reservation.cancel_reservation("SYN1A2B")
    assert result.status == "cancelled"
    assert result.requires_human_approval is True
    assert reservation.get_reservation("SYN1A2B") is None  # artik store'da yok


def test_cancel_reservation_refund_reflects_fare_type():
    # SYN3C4D: cabin_class=business, fare_type=Business -> yuksek iade orani
    result = reservation.cancel_reservation("SYN3C4D")
    assert result.refund_amount_estimate_try > 0
    # SYN5E6F: fare_type=EcoFly -> mock tabloda %0 iade
    result2 = reservation.cancel_reservation("SYN5E6F")
    assert result2.refund_amount_estimate_try == 0.0


def test_change_flight_date_updates_store():
    result = reservation.change_flight_date("SYN1A2B", "2026-09-01")
    assert result.old_date == "2026-08-03"
    assert result.new_date == "2026-09-01"
    assert reservation.get_reservation("SYN1A2B").date == "2026-09-01"


def test_add_baggage_is_not_critical():
    result = reservation.add_baggage("SYN1A2B", 2)
    assert result.requires_human_approval is False
    assert result.new_baggage_pieces == 3  # 1 (mevcut) + 2
    assert result.estimated_fee_try == 180.0  # 2 * 90


# --- checkin ---

def test_checkin_not_yet_open_when_far_from_flight_date():
    # SYN1A2B: ucus tarihi 2026-08-03, "bugun" 2026-07-30 -> 4 gun var
    result = checkin.check_in("SYN1A2B", current_date="2026-07-30")
    assert result.status == "not_yet_open"


def test_checkin_success_day_before_flight():
    # SYN1A2B: ucus tarihi 2026-08-03, "bugun" 2026-08-02 -> 1 gun var, pencere acik
    result = checkin.check_in("SYN1A2B", current_date="2026-08-02")
    assert result.status == "success"
    assert result.boarding_pass.seat == "14C"


def test_checkin_already_checked_in():
    # SYN5E6F mock veride checked_in=true olarak geliyor
    result = checkin.check_in("SYN5E6F", current_date="2026-07-29")
    assert result.status == "already_checked_in"


def test_checkin_window_closed_for_past_flight():
    result = checkin.check_in("SYN1A2B", current_date="2026-08-10")
    assert result.status == "window_closed"


def test_checkin_unknown_pnr_returns_none():
    assert checkin.check_in("YOKPNR", current_date="2026-07-30") is None


# --- FastAPI HTTP katmani (None -> 404 cevirisi dogru mu) ---

client = TestClient(app)


def test_http_flight_status_404_for_unknown_flight():
    response = client.get("/tools/flights/TK9999/status")
    assert response.status_code == 404


def test_http_reservation_lookup_success():
    response = client.get("/tools/reservations/SYN1A2B")
    assert response.status_code == 200
    assert response.json()["passenger_name"] == "Ayşe Yılmaz"


def test_http_cancel_reservation_404_for_unknown_pnr():
    response = client.post("/tools/reservations/YOKPNR/cancel")
    assert response.status_code == 404


def test_http_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
