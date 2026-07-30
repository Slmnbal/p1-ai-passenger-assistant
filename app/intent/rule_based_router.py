"""Kural/keyword tabanlı intent router — Adım 4'ün baseline'ı.

Data Scientist yaklaşımının gereği olarak (bkz. p1_proje_plani.md, "önce basit baseline,
sonra fine-tune" ilkesi — RAG'de TF-IDF/embedding karşılaştırmasında da aynı desen
kullanıldı), sıralı öncelikli kurallardan oluşur: ilk eşleşen kural kazanır. Bu, HF
fine-tuned modelin karşılaştırılacağı referans noktasıdır (bkz. data/intent/eda.py'deki
bulgu: "belirsiz_acikliga_kavusturma" sınıfı en kırılgan olanı — kurallar bunu en son,
"fallback" olarak ele alır).
"""

from __future__ import annotations

import re

INTENTS = [
    "kapsam_disi",
    "checkin_talebi",
    "rezervasyon_islem_talebi",
    "ucus_sorgulama",
    "belirsiz_acikliga_kavusturma",
    "politika_bilgi_sorgusu",
]

_KAPSAM_DISI_KEYWORDS = [
    "kargo", "kabin memuru", "hisse senedi", "pilot olmak", "teknik bakım",
    "yatırımcı", "filo", "sürdürülebilirlik", "turkish cargo", "staj",
    "hava durumu", "restoran", "şaka", "kredi kartı borc", "genel merkez",
    "iş ortaklığı", "basın bülten", "reklam ver", "sosyal medya hesab",
    "uçak motoru", "kariyer tavsiye", "sponsorluk", "simülatör deneyimi",
    "capital of", "sell tickets", "moon",
]

_CHECKIN_KEYWORDS = ["check-in", "check in", "biniş kart", "kiosk", "boarding pass"]

_ACTION_VERBS = [
    "istiyorum", "isterim", "ister misiniz", "edebilir miyim", "alabilir miyim",
    "yapabilir miyim", "want to", "need to", "can i", "please",
]
_ISLEM_KEYWORDS = [
    "iptal", "değiştir", "iade", "yükselt", "satın al", "ekle", "düzelt",
    "cancel", "refund", "change my", "erteleyebilir",
]

_UCUS_KEYWORDS = [
    "rötar", "gecikme var", "durumu", "kapı numarası", "kapıdan kalkıyor",
    "kaçta", "ne zaman iniyor", "ne zaman kalkıyor", "hangi bantta", "pnr",
    "uçuş numaras", "sefer sayılı", "flight status", "is my flight", "depart",
    "boş koltuk var", "kaç uçuş var", "hangi terminalden",
]

_BELIRSIZ_KEYWORDS = [
    "yardım lazım", "bir sorum var", "ne yapmalıyım", "emin değilim",
    "uygun mudur", "ücretsiz mi bu", "bu durumda", "hakkım var mı", "kurallar nedir",
]


def _contains_any(text_lower: str, keywords: list[str]) -> bool:
    return any(kw in text_lower for kw in keywords)


def classify(text: str) -> str:
    """Metni sıralı kurallarla bir intent'e atar. İlk eşleşen kural kazanır."""
    t = text.lower()

    if _contains_any(t, _KAPSAM_DISI_KEYWORDS):
        return "kapsam_disi"

    if _contains_any(t, _CHECKIN_KEYWORDS):
        return "checkin_talebi"

    if _contains_any(t, _ISLEM_KEYWORDS) and (_contains_any(t, _ACTION_VERBS) or True):
        return "rezervasyon_islem_talebi"

    if _contains_any(t, _UCUS_KEYWORDS) or re.search(r"\bthy\d+\b|\btk\d+\b", t):
        return "ucus_sorgulama"

    if _contains_any(t, _BELIRSIZ_KEYWORDS) or len(t.split()) <= 3:
        return "belirsiz_acikliga_kavusturma"

    return "politika_bilgi_sorgusu"
