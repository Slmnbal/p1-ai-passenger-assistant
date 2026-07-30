"""In-memory mock veri deposu — Adim 5'in tum tool'lari (ucus, rezervasyon, check-in)
bu moduldeki AYNI veriyi okur/yazar.

Neden ortak bir depo: check-in ve rezervasyon degisikligi ayni PNR uzerinde calisiyor;
her modul kendi JSON kopyasini ayri yuklerse (biri check-in yapar, digeri hala eski
"checked_in: false" gorur gibi) tutarsizlik cikar. Tek kaynak (single source of truth)
bunu mimari olarak imkansiz kilar.

Neden bellek-ici (in-memory) ve gercek veritabani degil: Bu mock/demo bir tool katmani
(bkz. p1_proje_plani.md Adim 5). FastAPI sureci ayakta oldugu surece degisiklikler
kalicidir (ayni PNR'i iki kere sorgulayinca ikinci sorguda guncel hali gorulur); surec
yeniden baslatilinca JSON dosyalarindan sifirdan yuklenir. Gercek bir veritabani (Adim 10+)
sonraki bir asamanin isi.
"""

from __future__ import annotations

import json
from pathlib import Path

_FLIGHTS_PATH = Path("data/mock_flights.json")
_RESERVATIONS_PATH = Path("data/mock_reservations.json")


def _load_flights() -> list[dict]:
    with open(_FLIGHTS_PATH, encoding="utf-8") as f:
        return json.load(f)["flights"]


def _load_reservations() -> dict[str, dict]:
    with open(_RESERVATIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # PNR'a gore anahtarlanmis sozluk: O(1) lookup, listede dongu gerekmez.
    return {r["pnr"]: dict(r) for r in data["reservations"]}


FLIGHTS: list[dict] = _load_flights()
RESERVATIONS: dict[str, dict] = _load_reservations()


def reset() -> None:
    """Testler icin: depoyu JSON dosyalarindaki orijinal haline sifirlar.

    Neden gerekli: pytest'te bir test PNR'i iptal ederse, sonraki test onu "iptal
    edilmemis" olarak bulmayi bekleyebilir. Testler birbirinden bagimsiz (izole)
    olmali — bu fonksiyon her testin kendi temiz durumdan baslamasini saglar.
    """
    global RESERVATIONS
    RESERVATIONS = _load_reservations()
