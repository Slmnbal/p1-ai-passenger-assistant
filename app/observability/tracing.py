"""Langfuse tabanlı LLM/agent trace — Adım 9.

Her `graph.run()` çağrısı bir Langfuse "trace"i olur (id = `request_id`, bkz. state.py);
her agent node'u (planning/retrieval/verification/tools/guardrail) bu trace altında bir
"span", her gerçek LLM çağrısı (Ollama — retrieval_agent'ın cevap üretimi,
grounding.py'nin claim decomposition/verification çağrıları) bir "generation" olarak
kaydedilir. Token/latency Langfuse UI'da otomatik görünür.

Neden düşük seviyeli client (`@observe` decorator DEĞİL): LangGraph node'ları birer
sözlük döndüren fonksiyonlar; decorator'ın varsaydığı "fonksiyonun kendi dönüş değeri
= LLM çıktısı" kalıbına tam uymuyor. Span/generation'ı elle `trace_id` ile açıp
kapatmak, hangi node'un ne kadar sürdüğünü (latency) ve LLM çağrılarının token/maliyet
bilgisini ayrı ayrı, doğru şekilde raporlamamızı sağlıyor.

`get_client()` bir lazy singleton (bkz. `policy_lookup.get_retriever()` ile aynı desen)
— her çağrıda yeni bir Langfuse client açmamak için.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from langfuse import Langfuse

_client: Langfuse | None = None


def get_client() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-p1-local-dev-key"),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-p1-local-dev-secret"),
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        )
    return _client


def start_trace(request_id: str, user_message: str) -> None:
    try:
        get_client().trace(id=request_id, name="p1-chat-turn", input=user_message)
    except Exception:
        pass  # Langfuse ayakta değilse trace atlanır, agent akışı asla bunun yüzünden kırılmaz


def log_span(request_id: str, name: str, start_time, end_time, input_data=None, output_data=None) -> None:
    try:
        get_client().span(
            trace_id=request_id,
            name=name,
            start_time=start_time,
            end_time=end_time,
            input=input_data,
            output=output_data,
        )
    except Exception:
        pass


def log_generation(
    request_id: str,
    name: str,
    model: str,
    start_time,
    end_time,
    input_data=None,
    output_data=None,
    usage: dict | None = None,
) -> None:
    try:
        get_client().generation(
            trace_id=request_id,
            name=name,
            model=model,
            start_time=start_time,
            end_time=end_time,
            input=input_data,
            output=output_data,
            usage=usage,
        )
    except Exception:
        pass


def now():
    return datetime.now(timezone.utc)


def flush() -> None:
    """Kısa ömürlü script/test çalıştırmalarında arka plan kuyruğunun gönderildiğinden
    emin olmak için — Langfuse SDK'sı olayları batch'leyip arka planda gönderiyor."""
    try:
        get_client().flush()
    except Exception:
        pass
