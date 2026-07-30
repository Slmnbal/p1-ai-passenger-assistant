"""Check-in mock araci.

`current_date` parametresi disaridan verilir, `datetime.now()` dogrudan kullanilmaz.
Neden: check-in penceresi mock rezervasyon tarihlerine (data/mock_reservations.json)
gore hesaplaniyor; kod dogrudan sistem saatine bagli olsaydi, bu sabit mock tarihler
zamanla "gecmiste" kalir ve bugun gecen bir test birkac ay sonra sessizce bozulurdu.
"Su anki zaman"i disaridan enjekte etmek (parametre olarak vermek), testlerin zamana
bagimli olmadan, her zaman ayni sonucu uretmesini saglar — gercek kullanimda ise
`current_date` verilmezse varsayilan olarak bugunun tarihi kullanilir.

Pencere kurali (bkz. data/policies/checkin_reservation_policy.md — gercek online
check-in kurali "24 saat-90 dakika once"dir; burada sadece TARIH bilgimiz oldugu icin
gunluk cozunurlukte basitlestirildi): ucus tarihine 1 gunden fazla varsa kapali,
ucus tarihi gecmisse kapali, aksi halde acik.
"""

from __future__ import annotations

from datetime import date, datetime

from app.tools import store
from app.tools.flight_search import mock_gate
from app.tools.schemas import BoardingPass, CheckInResponse

_BOARDING_TIME_OFFSET_MIN = 45  # bkz. boarding_gate_security_policy.md: genel kural 20-30 dk, mock icin sabit


def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def check_in(pnr: str, current_date: str | None = None) -> CheckInResponse | None:
    record = store.RESERVATIONS.get(pnr)
    if record is None:
        return None

    today = _parse(current_date) if current_date else date.today()
    flight_date = _parse(record["date"])
    days_until = (flight_date - today).days

    if record["checked_in"]:
        return CheckInResponse(status="already_checked_in", message="Bu rezervasyon için check-in zaten yapılmış.")

    if days_until < 0:
        return CheckInResponse(status="window_closed", message="Uçuş tarihi geçmiş, check-in penceresi kapandı.")

    if days_until > 1:
        return CheckInResponse(
            status="not_yet_open",
            message=f"Check-in henüz açık değil ({days_until} gün var). Uçuştan 1 gün önce açılır.",
        )

    record["checked_in"] = True
    boarding_pass = BoardingPass(
        pnr=pnr,
        passenger_name=record["passenger_name"],
        flight_no=record["flight_no"],
        seat=record["seat"],
        gate=mock_gate(record["flight_no"]),
        boarding_time=f"Kalkıştan {_BOARDING_TIME_OFFSET_MIN} dakika önce",
    )
    return CheckInResponse(status="success", boarding_pass=boarding_pass, message="Check-in tamamlandı.")
