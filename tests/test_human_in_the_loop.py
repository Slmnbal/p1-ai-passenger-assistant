"""Adım 8 human-in-the-loop onay katmanı için testler.

`reset_store`/`reset_queue` her testten önce çalışır (autouse) — Adım 5'teki
`test_tools.py`'deki izolasyon gerekçesiyle aynı: bir test bir PNR'ı iptal ederse,
sonraki test aynı PNR'ı hâlâ "aktif" bulmayı bekleyebilir.
"""

from __future__ import annotations

import pytest

from app.human_in_the_loop import approval_flow, approval_queue
from app.tools import store


@pytest.fixture(autouse=True)
def reset_state():
    store.reset()
    approval_queue.reset()
    yield


def test_submit_creates_pending_request_without_mutating_store():
    request = approval_queue.submit("cancel", "SYN3C4D", {})
    assert request.status == "pending"
    assert store.RESERVATIONS.get("SYN3C4D") is not None


def test_list_pending_returns_only_pending_requests():
    r1 = approval_queue.submit("cancel", "SYN3C4D", {})
    r2 = approval_queue.submit("change_date", "SYN5E6F", {"new_date": "2026-09-01"})
    approval_flow.approve(r1.id)

    pending_ids = {r.id for r in approval_queue.list_pending()}
    assert pending_ids == {r2.id}


def test_approve_cancel_executes_deferred_mutation():
    request = approval_queue.submit("cancel", "SYN3C4D", {})
    approved = approval_flow.approve(request.id)

    assert approved.status == "approved"
    assert approved.result["status"] == "cancelled"
    assert store.RESERVATIONS.get("SYN3C4D") is None  # şimdi gerçekten iptal edildi


def test_approve_change_date_executes_deferred_mutation():
    request = approval_queue.submit("change_date", "SYN5E6F", {"new_date": "2026-09-01"})
    approved = approval_flow.approve(request.id)

    assert approved.result["new_date"] == "2026-09-01"
    assert store.RESERVATIONS["SYN5E6F"]["date"] == "2026-09-01"


def test_reject_does_not_mutate_store():
    request = approval_queue.submit("cancel", "SYN3C4D", {})
    rejected = approval_flow.reject(request.id, reason="müşteri vazgeçti")

    assert rejected.status == "rejected"
    assert rejected.rejection_reason == "müşteri vazgeçti"
    assert store.RESERVATIONS.get("SYN3C4D") is not None


def test_approve_already_resolved_request_raises():
    request = approval_queue.submit("cancel", "SYN3C4D", {})
    approval_flow.approve(request.id)

    with pytest.raises(ValueError, match="sonuçlandırılmış"):
        approval_flow.approve(request.id)


def test_reject_already_resolved_request_raises():
    request = approval_queue.submit("cancel", "SYN3C4D", {})
    approval_flow.reject(request.id)

    with pytest.raises(ValueError, match="sonuçlandırılmış"):
        approval_flow.reject(request.id)


def test_approve_unknown_request_raises():
    with pytest.raises(ValueError, match="bulunamadı"):
        approval_flow.approve("does-not-exist")


def test_approve_nonexistent_reservation_result_reflects_not_found():
    request = approval_queue.submit("cancel", "YOKPNR", {})
    approved = approval_flow.approve(request.id)
    assert approved.result == {"found": False}
