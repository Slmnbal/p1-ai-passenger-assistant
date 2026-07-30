"""Adım 11 — segment analizi + iş KPI'ları.

`evaluation/e2e_results.json`'daki ham sonuçlardan (bkz. run_e2e_evaluation.py):
- Kategori/dil/mesaj-uzunluğu segmentlerine göre başarı oranı (bias/fairness kontrolü —
  bkz. Adım 4'ün segment analizi deseni, aynı yaklaşım burada uçtan uca akışa uygulandı)
- İş KPI'ları: ilk temasta çözüm, yanlış yönlendirme, insan devri, işlem tamamlama oranı

Çalıştırma: PYTHONPATH=. python evaluation/analyze_e2e_results.py
"""

from __future__ import annotations

import json
from collections import defaultdict

_ACTION_INTENTS = {"rezervasyon_islem_talebi", "checkin_talebi"}
_RESOLVED_OUTCOMES = {"grounded_answer", "tool_success"}
_HANDOFF_OUTCOMES = {"clarification", "approval_submitted"}


def _word_count_bucket(text: str) -> str:
    n = len(text.split())
    if n <= 4:
        return "kisa (<=4 kelime)"
    if n <= 10:
        return "orta (5-10 kelime)"
    return "uzun (>10 kelime)"


def _segment_report(results: list[dict], key_fn, title: str) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        buckets[key_fn(r)].append(r)

    print(f"\n=== {title} ===")
    report = {}
    for key in sorted(buckets):
        items = buckets[key]
        passed = sum(1 for r in items if r["passed"])
        rate = passed / len(items)
        report[key] = {"n": len(items), "passed": passed, "pass_rate": round(rate, 3)}
        print(f"  {key:28s} n={len(items):3d}  geçen={passed:3d}  oran={rate:.1%}")
    return report


def _business_kpis(results: list[dict]) -> dict:
    n = len(results)
    resolved = sum(1 for r in results if r["actual_outcome"] in _RESOLVED_OUTCOMES)
    misdirected = sum(
        1 for r in results
        if r["expected_intent"] is not None and r["actual_intent"] != r["expected_intent"]
    )
    handoff = sum(1 for r in results if r["actual_outcome"] in _HANDOFF_OUTCOMES)

    action_scenarios = [r for r in results if r["expected_intent"] in _ACTION_INTENTS]
    action_completed = sum(
        1 for r in action_scenarios
        if r["actual_outcome"] in ("tool_success", "approval_submitted", "not_found")
        # "not_found" da dahil çünkü PNR gerçekten yokken doğru şekilde "bulunamadı"
        # DEMEK zaten doğru TAMAMLANMIŞ bir işlemdir (yanlış bir varsayımda bulunmadı).
    )

    kpis = {
        "ilk_temasta_cozum_orani": round(resolved / n, 3),
        "yanlis_yonlendirme_orani": round(misdirected / n, 3),
        "insan_devri_orani": round(handoff / n, 3),
        "islem_tamamlama_orani": (
            round(action_completed / len(action_scenarios), 3) if action_scenarios else None
        ),
        "n_toplam": n,
        "n_islem_senaryosu": len(action_scenarios),
    }

    print("\n=== İş KPI'ları ===")
    print(f"  İlk temasta çözüm oranı : {kpis['ilk_temasta_cozum_orani']:.1%}  (n={n})")
    print(f"  Yanlış yönlendirme oranı: {kpis['yanlis_yonlendirme_orani']:.1%}")
    print(f"  İnsan devri oranı       : {kpis['insan_devri_orani']:.1%}")
    if action_scenarios:
        print(f"  İşlem tamamlama oranı   : {kpis['islem_tamamlama_orani']:.1%}  (n={len(action_scenarios)})")

    return kpis


def _known_error_pattern_report(results: list[dict]) -> dict:
    pattern_categories = ["ambiguous", "conflicting_source", "out_of_scope"]
    print("\n=== Bilinen hata kalıpları regresyon raporu (İlke 4) ===")
    report = {}
    for cat in pattern_categories:
        items = [r for r in results if r["category"] == cat]
        if not items:
            continue
        passed = sum(1 for r in items if r["passed"])
        rate = passed / len(items)
        report[cat] = {"n": len(items), "passed": passed, "pass_rate": round(rate, 3)}
        print(f"  {cat:20s} n={len(items):3d}  geçen={passed:3d}  oran={rate:.1%}")
        for r in items:
            if not r["passed"]:
                print(f"      FAIL {r['id']}: {r['error_category']} — {r['text'][:60]}")
    return report


def main() -> None:
    with open("evaluation/e2e_results.json", encoding="utf-8") as f:
        results = json.load(f)

    overall_passed = sum(1 for r in results if r["passed"])
    print(f"=== Genel sonuç: {overall_passed}/{len(results)} ({overall_passed / len(results):.1%}) ===")

    by_category = _segment_report(results, lambda r: r["category"], "Kategori bazlı başarı")
    by_language = _segment_report(results, lambda r: r["language"], "Dil bazlı başarı (bias/fairness)")
    by_length = _segment_report(
        results, lambda r: _word_count_bucket(r["text"]), "Mesaj uzunluğu bazlı başarı"
    )
    kpis = _business_kpis(results)
    error_patterns = _known_error_pattern_report(results)

    error_categories = defaultdict(int)
    for r in results:
        if r["error_category"]:
            error_categories[r["error_category"]] += 1
    if error_categories:
        print("\n=== Hata kategorisi dağılımı ===")
        for cat, count in sorted(error_categories.items(), key=lambda x: -x[1]):
            print(f"  {cat:30s} {count}")

    analysis = {
        "overall": {"n": len(results), "passed": overall_passed, "pass_rate": round(overall_passed / len(results), 3)},
        "by_category": by_category,
        "by_language": by_language,
        "by_message_length": by_length,
        "business_kpis": kpis,
        "known_error_patterns": error_patterns,
        "error_category_counts": dict(error_categories),
    }
    with open("evaluation/e2e_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
