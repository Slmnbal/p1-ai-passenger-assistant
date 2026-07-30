"""Kural tabanli intent router'i test setinde olcer: confusion matrix, sinif bazli
precision/recall/F1, Macro F1 (bkz. p1_proje_plani.md Adim 4, Ilke 2).

Calistirma: PYTHONPATH=. python evaluation/evaluate_intent_router.py
"""

from __future__ import annotations

import json

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from app.intent.rule_based_router import INTENTS, classify


def main() -> None:
    with open("data/intent/test.json", encoding="utf-8") as f:
        test_records = json.load(f)

    y_true = [r["intent"] for r in test_records]
    y_pred = [classify(r["text"]) for r in test_records]

    print(f"Test seti buyuklugu: {len(test_records)}\n")

    print("=== YANLIS SINIFLANDIRILAN ORNEKLER ===")
    for r, pred in zip(test_records, y_pred):
        if pred != r["intent"]:
            print(f"  GERCEK={r['intent']:<30} TAHMIN={pred:<30} \"{r['text']}\"")

    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_true, y_pred, labels=INTENTS)
    header = "".join(f"{i[:10]:>12}" for i in INTENTS)
    print(f"{'':>30}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>12}" for v in row)
        print(f"{INTENTS[i]:>30}{row_str}")

    print("\n=== SINIF BAZLI PRECISION / RECALL / F1 ===")
    print(classification_report(y_true, y_pred, labels=INTENTS, zero_division=0))

    macro_f1 = f1_score(y_true, y_pred, labels=INTENTS, average="macro", zero_division=0)
    print(f"Macro F1: {macro_f1:.3f}")

    # Kritik sinifa (rezervasyon_islem_talebi) ozel odak
    critical = "rezervasyon_islem_talebi"
    critical_indices = [i for i, t in enumerate(y_true) if t == critical]
    critical_correct = sum(1 for i in critical_indices if y_pred[i] == critical)
    print(f"\nKritik sinif ({critical}) dogruluk: {critical_correct}/{len(critical_indices)} "
          f"(%{critical_correct/len(critical_indices)*100:.1f})")

    results = {
        "y_true": y_true,
        "y_pred": y_pred,
        "macro_f1": macro_f1,
        "critical_class_accuracy": critical_correct / len(critical_indices),
    }
    with open("evaluation/intent_rule_based_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
