"""Prometheus metrikleri — Adım 9.

Kendi `CollectorRegistry`'mizi kullanıyoruz (varsayılan global registry yerine): pytest
testleri modülü birden çok kez import/reload edebilir, varsayılan registry'de "aynı
metrik iki kez kaydedilmeye çalışıldı" hatası (`Duplicated timeseries`) alma riskini
ortadan kaldırır — `app/rag/qdrant_retriever.py`'nin recreate=False ile tekrar
oluşturmayı önleme mantığıyla aynı aile: yeniden çalıştırmada çakışma yaratma.

Dört metrik, plan'ın "bir isteğin baştan sona izlenebilmesi" hedefine karşılık gelir:
- `p1_requests_total{intent}`: hangi intent ne sıklıkla geliyor
- `p1_guardrail_blocks_total{reason}`: guardrail hangi sebeple ne sıklıkla bloklu
- `p1_node_latency_seconds{node}`: her agent node'unun ne kadar sürdüğü
- `p1_approval_queue_pending`: o an bekleyen onay talebi sayısı (Adım 8'e bağlı)
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "p1_requests_total", "Toplam işlenen kullanıcı mesajı", ["intent"], registry=REGISTRY
)
GUARDRAIL_BLOCKS_TOTAL = Counter(
    "p1_guardrail_blocks_total",
    "Guardrail tarafından bloklanan istek sayısı",
    ["reason"],
    registry=REGISTRY,
)
NODE_LATENCY_SECONDS = Histogram(
    "p1_node_latency_seconds", "Agent node çalışma süresi (saniye)", ["node"], registry=REGISTRY
)
APPROVAL_QUEUE_PENDING = Gauge(
    "p1_approval_queue_pending", "O an bekleyen onay talebi sayısı", registry=REGISTRY
)


def render() -> bytes:
    return generate_latest(REGISTRY)
