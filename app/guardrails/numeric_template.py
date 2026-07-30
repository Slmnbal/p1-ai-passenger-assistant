"""Sayısal halüsinasyon kontrolü — Adım 7, İlke 4'ün "sayısal şablon enjeksiyonu" maddesi.

Plan aslında sayıların kaynaktan "çıkarılıp şablona yerleştirilmesini" (LLM'e hiç
yazdırmamayı) istiyor — bu, `tool_agent.py`'da zaten tam olarak yapılıyor (LLM YOK,
tool'un kendi şablonlanmış mesajı kullanılıyor). Ama `retrieval_agent.py`'nin serbest
metin ürettiği politika soruları için (soru açık uçlu, önceden hangi şablonun
kullanılacağı bilinemez) bu mümkün değil — bunun yerine POST-HOC bir doğrulama yapılıyor:
LLM'in cevabındaki HER sayı, retrieved chunk'ların GERÇEKTEN içerdiği bir sayı mı?

Basit ama deterministik (regex + küme karşılaştırması) — `grounding.py`'deki LLM-judge'ın
aksine burada bir dil modeli belirsizliği YOK, bu yüzden `judge_faithfulness`'ın aksine
bu kontrol graph.py'de BLOKLAYICI olarak kullanılıyor.

**Bilinen sınırlama:** Yalnızca SAYISAL halüsinasyonu yakalar — "ücretsiz sınırsız VIP
salon" gibi sayı içermeyen ama tamamen uydurma bir NİTELİKSEL iddiayı yakalamaz (bkz.
`grounding.py`'deki `require_sources` + kod incelemesinde belgelenen kalan boşluk).
Ayrıca küçük/genel sayılar (örn. "1", "2") tesadüfen bağlamda başka bir bağlamda da
geçebilir — bu, yanlış negatif riski yaratır ama yanlış pozitiften (doğru bir cevabı
gereksiz reddetmekten) daha az zararlıdır.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _normalize(n: str) -> str:
    n = n.replace(",", ".")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n or "0"


def _extract_numbers(text: str) -> set[str]:
    return {_normalize(n) for n in _NUMBER_RE.findall(text)}


def check_numeric_consistency(answer: str, sources: list[dict]) -> tuple[bool, list[str]]:
    """Cevaptaki her sayı en az bir kaynak metinde de geçiyor mu? (context_text birleşik)."""
    answer_numbers = _extract_numbers(answer)
    context_text = " ".join(s["text"] for s in sources)
    context_numbers = _extract_numbers(context_text)
    unsupported = sorted(answer_numbers - context_numbers)
    return len(unsupported) == 0, unsupported
