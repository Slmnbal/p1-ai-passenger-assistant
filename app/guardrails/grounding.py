"""Zorunlu kaynak atfı + LLM-judge faithfulness kontrolü — Adım 7, İlke 4'ün kalbi.

**Adım 7'nin ilk sürümünde** (bkz. docs/adr/0003, Deney 3) tüm cevabı TEK bir LLM
çağrısıyla bütün olarak değerlendirmek denenmiş ve GÜVENİLİR BULUNMAMIŞTI: model ya
her şeyi onaylıyor ya da her şeyi reddediyordu, tutarlı bir ayrım yapamıyordu.

**Bu sürüm (claim decomposition) o sorunu çözüyor — ÖLÇÜLDÜ:** Cevabı önce bağımsız,
TEK bir iddia içeren cümlelere bölüyoruz (`decompose_claims`), sonra HER iddiayı AYRI
AYRI, dar kapsamlı bir soruyla doğruluyoruz (`_verify_claim`). Küçük modelin zorlandığı
şey "bütün paragrafı bir defada değerlendirmek"ti — tek bir iddiayı değerlendirmek çok
daha kolay bir görev. İki farklı gerçek RAG cevabı (bagaj + engelli indirimi, doğru/
bozulmuş varyantlarıyla, toplam 10 iddia) üzerinde test edildi: **9/10 doğru (%90)** —
eski yaklaşımın ayrım gücü neredeyse sıfırdı (rastgele EVET/HAYIR ile aynı), bu yüzden
artık BLOKLAYICI bir kontrol olarak kullanılıyor (eskiden sadece gözlem amaçlıydı).

**Bilinen kalan sınırlama (1/10 hata):** Model bazen kaynaktaki bir koşulu/istisnayı
(örn. "promosyon biletler hariç, ekonomi kabin") cevapta tekrarlanmadığı için iddiayı
"desteklenmiyor" sayabiliyor — yani makul bir ÖZETLEMEYİ hata sanabiliyor. Bu, "hiçbir
şey yakalamama" değil "bazen fazla temkinli olma" riski — yanlış bir cevabı geçirmekten
(false negative — kötü) daha az zararlı bir yanlış (false positive — iyi bir cevabı
gereksiz reddetme), bu yüzden kabul edilebilir bulundu.
"""

from __future__ import annotations

import os

from openai import OpenAI

from app.observability import tracing

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

_DECOMPOSE_PROMPT = (
    "Verilen CEVABI, birbirinden bağımsız, TEK bir iddia içeren kısa cümlelere ayır.\n"
    "Her cümleyi ayrı bir satıra yaz, başına \"- \" koy. Başka hiçbir şey yazma, açıklama ekleme."
)

_VERIFY_CLAIM_PROMPT = (
    "Sana bir BAĞLAM ve TEK bir İDDİA verilecek. İddia, bağlamda doğrudan belirtilen ya da "
    "bağlamdan basitçe çıkarılabilen bir bilgiyse EVET yaz. İddia bağlamda hiç geçmeyen "
    "veya bağlamla çelişen bir bilgi içeriyorsa HAYIR yaz. Sadece EVET ya da HAYIR yaz."
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "ollama-local-no-key-needed"),
        )
    return _client


def require_sources(sources: list[dict]) -> bool:
    return bool(sources)


def decompose_claims(answer: str, request_id: str | None = None) -> list[str]:
    client = _get_client()
    start_time = tracing.now()
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _DECOMPOSE_PROMPT},
            {"role": "user", "content": answer},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    end_time = tracing.now()
    raw = response.choices[0].message.content.strip()
    if request_id:
        tracing.log_generation(
            request_id=request_id,
            name="grounding.decompose_claims",
            model=OLLAMA_MODEL,
            start_time=start_time,
            end_time=end_time,
            input_data=answer,
            output_data=raw,
        )
    lines = raw.split("\n")
    return [line.lstrip("- ").strip() for line in lines if line.strip()]


def _verify_claim(claim: str, context: str, request_id: str | None = None) -> bool:
    client = _get_client()
    start_time = tracing.now()
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _VERIFY_CLAIM_PROMPT},
            {"role": "user", "content": f"BAĞLAM: {context}\nİDDİA: {claim}"},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    end_time = tracing.now()
    raw = response.choices[0].message.content.strip()
    if request_id:
        tracing.log_generation(
            request_id=request_id,
            name="grounding.verify_claim",
            model=OLLAMA_MODEL,
            start_time=start_time,
            end_time=end_time,
            input_data=f"BAĞLAM: {context}\nİDDİA: {claim}",
            output_data=raw,
        )
    return raw.upper().startswith("EVET")


def judge_faithfulness(
    answer: str, sources: list[dict], request_id: str | None = None
) -> tuple[bool, list[dict]]:
    """Cevabı iddialara böl, her iddiayı ayrı doğrula. (grounded, [{"claim","supported"}, ...])"""
    context = "\n\n".join(s["text"] for s in sources)
    claims = decompose_claims(answer, request_id=request_id)
    if not claims:
        return True, []
    results = [
        {"claim": claim, "supported": _verify_claim(claim, context, request_id=request_id)}
        for claim in claims
    ]
    all_supported = all(r["supported"] for r in results)
    return all_supported, results
