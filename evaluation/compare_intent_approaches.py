"""Uc yaklasimi (kural tabanli, frozen-probe, full fine-tune) ayni test setinde karsilastirir.

Confusion matrix, sinif bazli precision/recall/F1, Macro F1, ROC-AUC (one-vs-rest,
sadece olasilik ureten modeller icin) ve kalibrasyon (full fine-tune icin).

Calistirma: PYTHONPATH=. python evaluation/compare_intent_approaches.py
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

from app.intent.rule_based_router import INTENTS, classify


def load_hf_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def macro_f1_of(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=INTENTS, average="macro", zero_division=0)


def main() -> None:
    with open("data/intent/test.json", encoding="utf-8") as f:
        test_records = json.load(f)
    y_true_text = [r["intent"] for r in test_records]

    # 1) Kural tabanli
    y_pred_rule = [classify(r["text"]) for r in test_records]

    # 2) Frozen-probe
    frozen = load_hf_results("models/intent_frozen/eval_results.json")

    # 3) Full fine-tune (10 epoch — en iyi denge noktasi)
    full10 = load_hf_results("models/intent_full_10ep/eval_results.json")
    full20 = load_hf_results("models/intent_full_20ep/eval_results.json")

    # 4) Dogrudan LLM routing (hipotezin karsilastirma kolu)
    with open("evaluation/intent_llm_routing_results.json", encoding="utf-8") as f:
        llm_data = json.load(f)
    llm_y_true = [r["y_true"] for r in llm_data["results"]]
    llm_y_pred = [r["y_pred"] for r in llm_data["results"]]

    approaches = {
        "Kural tabanli (baseline)": (y_true_text, y_pred_rule, None),
        "Frozen-probe (BERT donduruldu)": (frozen["y_true"], frozen["y_pred"], frozen["probs"]),
        "Full fine-tune (10 epoch)": (full10["y_true"], full10["y_pred"], full10["probs"]),
        "Full fine-tune (20 epoch, asiri ogrenme)": (full20["y_true"], full20["y_pred"], full20["probs"]),
        "Dogrudan LLM routing (llama3.1:8b)": (llm_y_true, llm_y_pred, None),
    }

    print(f"{'Yaklasim':<42} {'Accuracy':>10} {'Macro F1':>10} {'ROC-AUC (OvR)':>15}")
    print("-" * 80)

    summary = {}
    for name, (yt, yp, probs) in approaches.items():
        acc = sum(1 for a, b in zip(yt, yp) if a == b) / len(yt)
        mf1 = macro_f1_of(yt, yp)

        roc_auc = None
        if probs is not None:
            y_true_idx = [INTENTS.index(v) for v in yt]
            y_true_onehot = np.eye(len(INTENTS))[y_true_idx]
            try:
                roc_auc = roc_auc_score(y_true_onehot, np.array(probs), average="macro", multi_class="ovr")
            except ValueError:
                roc_auc = float("nan")

        roc_str = f"{roc_auc:.3f}" if roc_auc is not None else "n/a"
        print(f"{name:<42} {acc:>10.3f} {mf1:>10.3f} {roc_str:>15}")
        summary[name] = {"accuracy": acc, "macro_f1": mf1, "roc_auc": roc_auc}

    print("\n=== Kritik sinif (rezervasyon_islem_talebi) dogrulugu ===")
    for name, (yt, yp, _) in approaches.items():
        idx = [i for i, v in enumerate(yt) if v == "rezervasyon_islem_talebi"]
        correct = sum(1 for i in idx if yp[i] == "rezervasyon_islem_talebi")
        print(f"  {name}: {correct}/{len(idx)} (%{correct/len(idx)*100:.1f})")

    # Kalibrasyon: full fine-tune (10 epoch) icin - tahmin edilen en yuksek olasilik
    # ile gercek dogruluk arasindaki iliski (10 bin'e ayirarak)
    print("\n=== Kalibrasyon (Full fine-tune 10 epoch) ===")
    probs_arr = np.array(full10["probs"])
    max_probs = probs_arr.max(axis=1)
    pred_idx = probs_arr.argmax(axis=1)
    y_true_idx = [INTENTS.index(v) for v in full10["y_true"]]
    correct = np.array([p == t for p, t in zip(pred_idx, y_true_idx)])

    bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
    for i in range(len(bins) - 1):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = max_probs[mask].mean()
            print(f"  Güven [{bins[i]:.1f}-{bins[i+1]:.1f}): n={mask.sum():<3} "
                  f"ort.güven={bin_conf:.3f} gerçek_doğruluk={bin_acc:.3f}")

    with open("evaluation/intent_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
