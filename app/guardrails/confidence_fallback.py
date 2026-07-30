"""Güven eşiği fallback'i — Adım 7, ama önce ÖLÇÜLDÜ, sonra yazıldı.

İlke 4: "Retrieval benzerlik skoru veya intent güveni eşiğin altındaysa sistem tahmin
üretmez." Bunu kod olarak yazmadan önce, `evaluation/eval_results_ctx.json`'daki 72
sorunun top1_score'unu kategoriye göre ölçtük:

    kontrol           n=51  min=0.288  p25=0.629  mean=0.687  max=0.899
    kafa_karistirici  n=8   min=0.631  p25=0.659  mean=0.716  max=0.856
    kapsam_disi       n=8   min=0.298  p25=0.547  mean=0.575  max=0.800
    es_anlamli        n=5   min=0.471  p25=0.516  mean=0.534  max=0.582

**Sonuç: retrieval skoru için HİÇBİR sabit eşik işe yaramaz.** kontrol'ün min'i (0.288)
kapsam_dışı'nın max'ından (0.800) DAHA DÜŞÜK — yani dağılımlar tamamen iç içe. Bu,
ADR-0002 Deney 2'nin (kelime örtüşmesi sinyali de aynı şekilde ayırt edemiyordu) bir kez
daha doğrulanması. Bu yüzden burada bir retrieval-skor-eşiği YOK — olsaydı ya çoğu meşru
soruyu reddederdi ya da hiçbir kapsam dışı soruyu yakalamazdı. Bkz. `docs/adr/0003-...md`.

Intent güven skoru biraz daha kullanışlı ama yine de zayıf bir sinyal (bkz.
`planning_agent.py`'deki ölçüm: thr=0.3→0.8 arası tutma oranı düşerken doğruluk sadece
%74.5→%81.1'e çıkıyor — net ama küçük bir iyileşme). Bu yüzden burada SERT bir blokaj
değil, yalnızca gözlemlenebilirlik bayrağı (Adım 9'da audit log'a düşecek) üretiliyor.

Asıl "düşük güvende tahmin üretme" ilkesi, retrieval skoruna değil, CEVAP ÜRETİLDİKTEN
SONRA kaynakla gerçekten örtüşüp örtüşmediğine bakan `grounding.py` + `numeric_template.py`
kontrollerine dayanıyor — Adım 6'da canlı bir örnekle (intent modelinin %50.6 güvenle
yanlış sınıflandırdığı ama RAG/verification katmanının yakaladığı "filodaki uçak tipleri"
sorusu) bunun skor-eşiğinden daha güvenilir olduğu zaten kanıtlandı.
"""

from __future__ import annotations

INTENT_CONFIDENCE_OBSERVE_FLOOR = 0.4


def flag_low_confidence_intent(intent_confidence: float) -> bool:
    """Sert bir blokaj değil — yalnızca audit/log için bir gözlem bayrağı üretir."""
    return intent_confidence < INTENT_CONFIDENCE_OBSERVE_FLOOR
