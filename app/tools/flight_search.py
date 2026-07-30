"""Ucus arama/durum sorgulama mock araci.

Gercek bir ucus takip sistemine baglanmiyoruz (boyle bir erisimimiz yok ve olsa bile
ucretli olurdu — bkz. proje kisiti "para harcanmayacak"). Bunun yerine data/mock_flights.json
icindeki gercek THY rota agina dayanan (ama uydurma saat/durum tasiyan) sabit veriden,
HER SORGUDA AYNI SONUCU VEREN (deterministik) bir "sahte durum" turetiyoruz.

Neden deterministik (rastgele degil): Ayni ucus numarasi iki kere sorulunca farkli cevap
donerse hem kullanici deneyimi tutarsiz olur hem de pytest testleri kararsizlasir (bazen
gecer bazen kalir). Python'un `hash()` fonksiyonu yerine `hashlib` kullanildi cunku
`hash()` her Python calistirmasinda farkli tuz (salt) ile calisir — testler arasi
tutarlilik icin gercek bir kriptografik hash (md5) sart.
"""

from __future__ import annotations

import hashlib

from app.tools import store
from app.tools.schemas import FlightInfo, FlightSearchRequest, FlightSearchResponse


def _deterministic_seed(flight_no: str) -> int:
    """flight_no'dan, calistirmalar arasi degismeyen bir tam sayi uretir."""
    digest = hashlib.md5(flight_no.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _mock_status(flight_no: str) -> tuple[str, int]:
    """Deterministik sahte durum: %70 zamaninda, %20 gecikmeli, %10 iptal."""
    seed = _deterministic_seed(flight_no)
    bucket = seed % 10
    if bucket < 7:
        return "on_time", 0
    if bucket < 9:
        delay = 15 + (seed % 6) * 15  # 15, 30, 45, 60, 75, 90 dakikadan biri
        return "delayed", delay
    return "cancelled", 0


def mock_gate(flight_no: str) -> str:
    seed = _deterministic_seed(flight_no)
    letter = chr(ord("A") + seed % 6)  # A-F arasi pier
    number = 1 + seed % 30
    return f"{letter}{number}"


def search_flights(request: FlightSearchRequest) -> FlightSearchResponse:
    matches = [
        f for f in store.FLIGHTS
        if f["origin"] == request.origin and f["destination"] == request.destination
    ]

    results = []
    for f in matches:
        status, delay = _mock_status(f["flight_no"])
        results.append(FlightInfo(
            flight_no=f["flight_no"],
            origin=f["origin"],
            destination=f["destination"],
            typical_duration_min=f["typical_duration_min"],
            status=status,
            delay_minutes=delay,
            gate=mock_gate(f["flight_no"]) if status != "cancelled" else None,
        ))

    return FlightSearchResponse(query=request, flights=results)


def get_flight_status(flight_no: str) -> FlightInfo | None:
    match = next((f for f in store.FLIGHTS if f["flight_no"] == flight_no), None)
    if match is None:
        return None

    status, delay = _mock_status(flight_no)
    return FlightInfo(
        flight_no=match["flight_no"],
        origin=match["origin"],
        destination=match["destination"],
        typical_duration_min=match["typical_duration_min"],
        status=status,
        delay_minutes=delay,
        gate=mock_gate(flight_no) if status != "cancelled" else None,
    )
