"""Adim 4'te yapilan 5 intent siniflandirma denemesini MLflow'a kaydeder.

Not: Bu denemeler MLflow kurulmadan ONCE calistirildigi icin (once yontem calisiyor mu
diye dogrulandi, sonra izleme katmani eklendi — bkz. proje hafizasi
"professional-rigor-mentality": tek seferde tek yenilik), kayit geriye donuktur. Bundan
sonraki TUM egitim calismalari (train_classifier.py) canli olarak MLflow'a loglanacak.

Calistirma: PYTHONPATH=. python evaluation/log_experiments_to_mlflow.py
Not gerektirir: MLflow http://localhost:5001'de calisiyor olmali.
"""

from __future__ import annotations

import json

import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("P1-intent-classification")


def log_run(name: str, params: dict, metrics: dict, notes: str) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        mlflow.set_tag("logged_retroactively", "true")
        mlflow.set_tag("notes", notes)


def main() -> None:
    with open("evaluation/intent_comparison_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    with open("evaluation/intent_llm_routing_results.json", encoding="utf-8") as f:
        llm_data = json.load(f)
    llm_results = llm_data["results"]
    llm_correct = sum(1 for r in llm_results if r["y_true"] == r["y_pred"])

    with open("evaluation/intent_rule_based_results.json", encoding="utf-8") as f:
        rule_data = json.load(f)

    runs = [
        ("kural_tabanli_baseline",
         {"approach": "rule_based", "base_model": "n/a"},
         {"accuracy": summary["Kural tabanli (baseline)"]["accuracy"],
          "macro_f1": summary["Kural tabanli (baseline)"]["macro_f1"],
          "critical_class_accuracy": rule_data["critical_class_accuracy"]},
         "Keyword/regex tabanli, egitim gerektirmez."),

        ("frozen_probe_bert",
         {"approach": "frozen_probe", "base_model": "dbmdz/bert-base-turkish-cased",
          "epochs": 15, "learning_rate": 1e-3, "bert_frozen": True},
         {"accuracy": summary["Frozen-probe (BERT donduruldu)"]["accuracy"],
          "macro_f1": summary["Frozen-probe (BERT donduruldu)"]["macro_f1"],
          "roc_auc": summary["Frozen-probe (BERT donduruldu)"]["roc_auc"]},
         "BERT govdesi dondu, sadece siniflandirma katmani egitildi."),

        ("full_finetune_10epoch_SECILEN",
         {"approach": "full_finetune", "base_model": "dbmdz/bert-base-turkish-cased",
          "epochs": 10, "learning_rate": 5e-5, "bert_frozen": False},
         {"accuracy": summary["Full fine-tune (10 epoch)"]["accuracy"],
          "macro_f1": summary["Full fine-tune (10 epoch)"]["macro_f1"],
          "roc_auc": summary["Full fine-tune (10 epoch)"]["roc_auc"]},
         "SECILEN MODEL — en iyi denge noktasi (bkz. MODEL_CARD.md)."),

        ("full_finetune_20epoch_asiri_ogrenme",
         {"approach": "full_finetune", "base_model": "dbmdz/bert-base-turkish-cased",
          "epochs": 20, "learning_rate": 5e-5, "bert_frozen": False},
         {"accuracy": summary["Full fine-tune (20 epoch, asiri ogrenme)"]["accuracy"],
          "macro_f1": summary["Full fine-tune (20 epoch, asiri ogrenme)"]["macro_f1"],
          "roc_auc": summary["Full fine-tune (20 epoch, asiri ogrenme)"]["roc_auc"]},
         "Asiri ogrenme: train_loss ~0 ama eval_loss yukseldi, Macro F1 dustu."),

        ("llm_direct_routing_llama3.1",
         {"approach": "llm_prompt_routing", "base_model": "llama3.1:8b", "temperature": 0.0},
         {"accuracy": llm_correct / len(llm_results),
          "avg_latency_seconds": llm_data["elapsed_seconds"] / len(llm_results)},
         "Hipotez karsilastirma kolu: Ollama'ya prompt ile dogrudan siniflandirma."),
    ]

    for name, params, metrics, notes in runs:
        log_run(name, params, metrics, notes)
        print(f"Kaydedildi: {name}")

    print(f"\nMLflow UI: http://localhost:5001")


if __name__ == "__main__":
    main()
