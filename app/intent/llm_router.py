"""Dogrudan LLM routing — hipotezin karsilastirma kolu.

"Intent modeli + guven esigi, dogrudan LLM routing'e gore kritik taleplerde daha az
yanlis tool secer" hipotezini test etmek icin: kural tabanli/fine-tuned model yerine,
mesaji dogrudan Ollama'ya (yerel LLM) gonderip intent'i prompt ile sorduruyoruz. THY'nin
gercek OpenAI SDK + Ollama base_url deseni kullanilir (bkz. README, .env.example).
"""

from __future__ import annotations

import os

from openai import OpenAI

from app.intent.rule_based_router import INTENTS

_INTENT_DESCRIPTIONS = {
    "politika_bilgi_sorgusu": "Politika/kural bilgisi isteyen genel soru (bagaj hakkı, indirim, check-in kuralı vb. — eylem talebi yok, sadece bilgi soruyor)",
    "ucus_sorgulama": "Belirli bir uçuşun durumu/detayı hakkında salt-okunur sorgu (rötar mı, kaçta kalkıyor, hangi kapı)",
    "rezervasyon_islem_talebi": "Rezervasyon/bilette DEĞİŞİKLİK yapma talebi (iptal, tarih değişikliği, iade, yükseltme, ekstra satın alma)",
    "checkin_talebi": "Check-in işlemi yapma talebi",
    "belirsiz_acikliga_kavusturma": "Eksik bağlamlı, birden fazla yoruma açık veya çok kısa/genel mesaj (netleştirme gerekir)",
    "kapsam_disi": "Havayolu yolcu hizmetleriyle ilgisi olmayan istek (kargo, işe alım, yatırımcı ilişkileri vb.)",
}

_SYSTEM_PROMPT = (
    "Sen bir havayolu yolcu asistanının intent siniflandirma modulusun. "
    "Kullanicidan gelen mesaji asagidaki 6 kategoriden TAM OLARAK birine ata. "
    "Sadece kategori adini yaz, baska hicbir sey yazma (aciklama, noktalama, tirnak yok).\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in _INTENT_DESCRIPTIONS.items())
)


def classify_with_llm(text: str, client: OpenAI, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Mesaj: \"{text}\"\n\nKategori:"},
        ],
        temperature=0.0,
        max_tokens=20,
    )
    raw = response.choices[0].message.content.strip().lower()

    for intent in INTENTS:
        if intent in raw:
            return intent
    return "politika_bilgi_sorgusu"  # eslesme yoksa en genel sinifa dus (kural router'daki fallback ile tutarli)


def build_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "ollama-local-no-key-needed"),
    )
