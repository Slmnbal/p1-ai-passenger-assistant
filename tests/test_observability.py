"""Adım 9 gözlemlenebilirlik katmanı için testler.

`structured_logging`/`audit_log`/`metrics` tamamen deterministik ve yerel (dış servise
ihtiyaç yok). `TestLiveTracing` sınıfı GERÇEK Langfuse'a bağlanır (docker-compose'daki
`langfuse-server` açık olmalı) — bir span/generation gönderip Langfuse'un public API'siyle
gerçekten kaydedildiğini doğrular; sadece "hata fırlatmadı" demek yeterli olmazdı, çünkü
`tracing.py`'deki tüm fonksiyonlar try/except ile hataları yutuyor (agent akışını Langfuse
kesintisinde kırmamak için) — bu yüzden canlı doğrulama şart.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.request
import uuid

import pytest

from app.observability import audit_log, metrics, tracing
from app.observability.structured_logging import JsonFormatter, get_logger, log_event


@pytest.fixture(autouse=True)
def reset_audit_log():
    audit_log.reset()
    yield
    audit_log.reset()


# --- structured_logging ---

def test_json_formatter_produces_valid_json_with_request_id():
    logger = get_logger("test")
    record = logger.makeRecord(
        "p1.test", logging.INFO, __file__, 0, "test mesajı", (), None
    )
    record.request_id = "abc123"
    record.extra_fields = {"foo": "bar"}

    formatted = JsonFormatter().format(record)
    payload = json.loads(formatted)

    assert payload["message"] == "test mesajı"
    assert payload["request_id"] == "abc123"
    assert payload["foo"] == "bar"
    assert payload["level"] == "INFO"


def test_log_event_helper_attaches_request_id_and_fields():
    """`caplog` yerine logger'a doğrudan bir handler ekliyoruz — `configure_logging()`
    (graph.py import edildiğinde çağrılıyor) `p1` logger'ında `propagate=False`
    ayarlıyor; bu, testin çalıştırılma sırasına göre caplog'un (root handler'a
    dayanan) bazen hiç kayıt görmemesine yol açardı. Doğrudan handler eklemek bu
    sıra-bağımlılığından bağımsız, güvenilir bir test yapıyor.

    Logger seviyesini de AÇIKÇA INFO'ya çekiyoruz: `configure_logging()` hiç
    çağrılmamışsa (örn. `pytest -m "not live"` ile graph.py'yi import eden Live
    testler devre dışı bırakıldığında) "p1" logger'ı varsayılan WARNING seviyesinde
    kalır ve INFO seviyeli bir log_event sessizce yutulurdu — bu da testin, hangi
    testlerin ondan ÖNCE çalıştığına bağlı, kırılgan (flaky) olmasına yol açardı."""
    logger = get_logger("test_log_event")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append
    logger.addHandler(handler)
    try:
        log_event(logger, logging.INFO, "olay oldu", request_id="req-1", extra_key="extra_val")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    assert len(captured) == 1
    assert captured[0].request_id == "req-1"
    assert captured[0].extra_fields == {"extra_key": "extra_val"}


# --- audit_log ---

def test_audit_log_record_and_read_all_roundtrip():
    entry = audit_log.record(actor="system", action="test_action", request_id="req-1", pnr="SYN1A2B")
    all_entries = audit_log.read_all()

    assert len(all_entries) == 1
    assert all_entries[0]["actor"] == "system"
    assert all_entries[0]["action"] == "test_action"
    assert all_entries[0]["pnr"] == "SYN1A2B"
    assert all_entries[0]["timestamp"] == entry["timestamp"]


def test_audit_log_reset_clears_file():
    audit_log.record(actor="system", action="x")
    audit_log.reset()
    assert audit_log.read_all() == []


def test_audit_log_read_all_returns_empty_list_when_no_file():
    assert audit_log.read_all() == []


# --- metrics ---

def test_metrics_render_contains_all_defined_metric_names():
    metrics.REQUESTS_TOTAL.labels(intent="test_intent").inc()
    metrics.GUARDRAIL_BLOCKS_TOTAL.labels(reason="test_reason").inc()
    metrics.NODE_LATENCY_SECONDS.labels(node="test_node").observe(0.5)
    metrics.APPROVAL_QUEUE_PENDING.set(2)

    rendered = metrics.render().decode("utf-8")

    assert "p1_requests_total" in rendered
    assert "p1_guardrail_blocks_total" in rendered
    assert "p1_node_latency_seconds" in rendered
    assert "p1_approval_queue_pending" in rendered
    assert 'intent="test_intent"' in rendered


def test_approval_queue_gauge_reflects_pending_count():
    from app.human_in_the_loop import approval_queue
    from app.tools import store

    store.reset()
    approval_queue.reset()
    assert metrics.APPROVAL_QUEUE_PENDING._value.get() == 0

    r1 = approval_queue.submit("cancel", "SYN3C4D", {})
    assert metrics.APPROVAL_QUEUE_PENDING._value.get() == 1

    approval_queue.submit("change_date", "SYN5E6F", {"new_date": "2026-09-01"})
    assert metrics.APPROVAL_QUEUE_PENDING._value.get() == 2

    r1.status = "approved"
    approval_queue.update(r1)
    assert metrics.APPROVAL_QUEUE_PENDING._value.get() == 1


# --- Langfuse (gerçek servis gerektirir) ---

@pytest.mark.live
class TestLiveTracing:
    def _fetch_trace(self, trace_id: str) -> dict:
        auth = base64.b64encode(b"pk-lf-p1-local-dev-key:sk-lf-p1-local-dev-secret").decode()
        req = urllib.request.Request(
            f"http://localhost:3000/api/public/traces/{trace_id}",
            headers={"Authorization": f"Basic {auth}"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def test_span_and_generation_are_actually_recorded_in_langfuse(self):
        request_id = str(uuid.uuid4())
        tracing.start_trace(request_id, "test mesajı")
        start = tracing.now()
        end = tracing.now()
        tracing.log_span(request_id, "test-span", start, end, input_data="in", output_data="out")
        tracing.log_generation(
            request_id, "test-generation", "llama3.1:8b", start, end,
            input_data="prompt", output_data="completion",
        )
        tracing.flush()
        time.sleep(1)  # Langfuse'un olayı işlemesi için kısa bir bekleme

        trace = self._fetch_trace(request_id)
        assert trace["id"] == request_id
        observation_names = {o["name"] for o in trace["observations"]}
        assert "test-span" in observation_names
        assert "test-generation" in observation_names
