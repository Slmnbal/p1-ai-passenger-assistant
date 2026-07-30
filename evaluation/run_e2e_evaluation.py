"""Adım 11 — uçtan uca senaryo değerlendirmesi.

`evaluation/e2e_scenarios.json`'daki her senaryoyu GERÇEK `graph.run()` ile (gerçek
Qdrant + Ollama + fine-tuned intent modeli + guardrail katmanı) çalıştırır — RAG'in
kendi başına test edilmesinden (Adım 3) farklı olarak, burada TÜM agent zincirinin
(planlama -> retrieval/tool -> guardrail) uçtan uca doğru çalışıp çalışmadığı ölçülüyor.

Her senaryo için ham sonuç `evaluation/e2e_results.json`'a yazılır; segment/KPI analizi
ayrı olarak `evaluation/analyze_e2e_results.py` ile bu dosyadan yapılır (ham veri ile
analiz mantığını ayırmak, analiz mantığını yeniden çalıştırmak için tüm senaryoları
tekrar tekrar agent'tan geçirme zorunluluğunu ortadan kaldırır — LLM çağrıları ucuz değil).

Çalıştırma: PYTHONPATH=. python evaluation/run_e2e_evaluation.py
"""

from __future__ import annotations

import json
import time

from app.agent.graph import run as run_agent
from app.human_in_the_loop import approval_queue
from app.tools import store


def _derive_outcome(final_state: dict) -> str:
    if final_state.get("needs_clarification"):
        return "clarification"
    if final_state.get("intent") == "kapsam_disi":
        return "blocked_out_of_scope"
    tool_result = final_state.get("tool_result") or {}
    if "approval_request" in tool_result:
        return "approval_submitted"
    if tool_result.get("found") is False:
        return "not_found"
    if tool_result:
        return "tool_success"
    # ÖNEMLİ: `blocked`, `grounded is True`'DAN ÖNCE kontrol ediliyor. Neden: verify_node
    # (Adım 6) bir cevabı "grounded=True" olarak işaretleyebilir ama output_guard (Adım 7)
    # SONRADAN daha sıkı bir kontrolle (claim decomposition/numeric) onu YİNE DE
    # bloklayıp final_response'u fallback mesajına çevirebilir — state'teki `grounded`
    # alanı bu geç müdahaleden sonra geriye alınmıyor (hâlâ True görünüyor). Bu sıralama
    # hatası ilk sürümde vardı (bkz. docs/adr/0004), bir senaryoyu (e002) yanlışlıkla
    # "grounded_answer" diye etiketlemişti — asıl final_response fallback mesajıydı.
    if final_state.get("blocked"):
        return "blocked_guardrail_or_injection"
    if final_state.get("grounded") is True:
        return "grounded_answer"
    if final_state.get("grounded") is False:
        # LLM'in kendisi "bulamadım" dedi (policy_verification_agent'ın literal-ifade
        # kontrolü) VEYA retry'lar tükenip fallback verildi — ikisi de aynı, dürüst
        # "cevap üretilemedi" sonucu; guardrail'in bloke ettiği durumdan (grounded=None
        # kalır, blocked=True) AYRI bir kategori.
        return "fallback_unknown"
    return "unknown"


def _error_category(scenario: dict, final_state: dict, actual_outcome: str) -> str | None:
    expected_intent = scenario.get("expected_intent")
    actual_intent = final_state.get("intent")
    if expected_intent is not None and actual_intent != expected_intent:
        return "yanlis_intent"

    if actual_outcome != scenario["expected_outcome"]:
        return "yanlis_outcome"

    expected_key = scenario.get("expected_tool_result_key")
    if expected_key:
        tool_result = final_state.get("tool_result") or {}
        if expected_key not in tool_result:
            return "yanlis_tool_secimi"

    expected_contains = scenario.get("expected_answer_contains")
    if expected_contains:
        answer = (final_state.get("final_response") or "").lower()
        if not any(s.lower() in answer for s in expected_contains):
            return "sayisal_veya_icerik_hatasi"

    return None


def run_evaluation(scenarios_path: str = "evaluation/e2e_scenarios.json") -> list[dict]:
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    results = []
    for i, scenario in enumerate(scenarios, start=1):
        store.reset()
        approval_queue.reset()

        start = time.monotonic()
        final_state = run_agent(scenario["text"])
        duration = time.monotonic() - start

        actual_outcome = _derive_outcome(final_state)
        error_category = _error_category(scenario, final_state, actual_outcome)

        result = {
            "id": scenario["id"],
            "text": scenario["text"],
            "category": scenario["category"],
            "language": scenario["language"],
            "expected_intent": scenario.get("expected_intent"),
            "actual_intent": final_state.get("intent"),
            "expected_outcome": scenario["expected_outcome"],
            "actual_outcome": actual_outcome,
            "passed": error_category is None,
            "error_category": error_category,
            "duration_seconds": round(duration, 2),
            "final_response": final_state.get("final_response"),
            "blocked": final_state.get("blocked", False),
        }
        results.append(result)
        status = "OK  " if result["passed"] else "FAIL"
        print(f"[{i:2d}/{len(scenarios)}] {status} {scenario['id']} ({duration:4.1f}s) {scenario['text'][:60]}")
        if error_category:
            print(f"          -> {error_category} | beklenen={scenario['expected_outcome']} gerçek={actual_outcome}")

    return results


if __name__ == "__main__":
    all_results = run_evaluation()
    with open("evaluation/e2e_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    passed = sum(1 for r in all_results if r["passed"])
    print(f"\n=== Toplam: {passed}/{len(all_results)} geçti ===")
