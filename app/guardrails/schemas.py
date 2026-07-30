"""Guardrail JSON şeması + tool_result üzerinde iş kuralı kontrolleri — Adım 7.

`GuardrailReport`, graph.py'nin output_guard node'unun ürettiği tek, yapılandırılmış
sonuç nesnesi: hangi kontrollerin çalıştığı, hangisinin geçtiği/geçmediği ve varsa
neden bloklandığı — Adım 9'daki audit log'un doğrudan yazabileceği şekilde tasarlandı.

İş kuralı kontrolleri BİLEREK dar kapsamlı: `app/tools/reservation.py` zaten
`requires_human_approval`'ı sabit bir iş kuralı olarak (koddan hesaplamadan) atıyor —
burada yapılan, o sabitin YANLIŞLIKLA (bug ya da ileride bir refactor'la) değişip
değişmediğini SINAMAK, yeni bir iş kuralı icat etmek değil. Aynı mantık `add_baggage`
için de geçerli (kritik olmadığı için approval GEREKMEZ, bunu da doğruluyoruz).

**Adım 8 notu:** `cancel`/`change_date` kontrolleri artık `tool_agent.py`'nin canlı
akışında hiç tetiklenmiyor (bu ikisi artık `approval_request` olarak kuyruğa giriyor,
bkz. `app/human_in_the_loop/`) — ama `approval_flow.approve()`'ın ürettiği gerçek
sonucu doğrulamak için hâlâ geçerli/testli kalıyorlar. `approval_request_is_pending`,
kuyruğa yeni giren bir talebin durumunun gerçekten "pending" olduğunu doğrular.
"""

from __future__ import annotations

from pydantic import BaseModel


class GuardrailReport(BaseModel):
    passed: bool
    checks: dict[str, bool]
    reason: str | None = None


def check_business_rules(intent: str, tool_result: dict) -> GuardrailReport:
    checks: dict[str, bool] = {}

    if "cancel" in tool_result:
        cancel = tool_result["cancel"]
        checks["cancel_status_correct"] = cancel.get("status") == "cancelled"
        checks["cancel_refund_non_negative"] = cancel.get("refund_amount_estimate_try", -1) >= 0
        checks["cancel_requires_human_approval"] = cancel.get("requires_human_approval") is True

    if "change_date" in tool_result:
        change = tool_result["change_date"]
        checks["change_date_requires_human_approval"] = change.get("requires_human_approval") is True

    if "add_baggage" in tool_result:
        baggage = tool_result["add_baggage"]
        checks["add_baggage_fee_non_negative"] = baggage.get("estimated_fee_try", -1) >= 0
        checks["add_baggage_no_approval_needed"] = baggage.get("requires_human_approval") is False

    if "approval_request" in tool_result:
        request = tool_result["approval_request"]
        checks["approval_request_is_pending"] = request.get("status") == "pending"
        checks["approval_request_action_is_critical"] = request.get("action") in ("cancel", "change_date")

    if "checkin" in tool_result:
        checkin = tool_result["checkin"]
        checks["checkin_status_known"] = checkin.get("status") in (
            "success",
            "already_checked_in",
            "not_yet_open",
            "window_closed",
        )

    if not checks:
        return GuardrailReport(passed=True, checks={}, reason=None)

    passed = all(checks.values())
    failed = [name for name, ok in checks.items() if not ok]
    reason = None if passed else f"İş kuralı ihlali: {', '.join(failed)}"
    return GuardrailReport(passed=passed, checks=checks, reason=reason)
