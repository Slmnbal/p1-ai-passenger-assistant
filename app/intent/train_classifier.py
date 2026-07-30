"""Intent siniflandirici egitimi — Adim 4.

Iki mod karsilastirilir (bkz. p1_proje_plani.md, Ilke 2 — "base model vs fine-tuned"):
- "frozen": dbmdz/bert-base-turkish-cased'in agirliklari dondurulur, sadece ustune
  eklenen siniflandirma katmani egitilir (klasik "linear probe" / feature extraction).
  Bu, "modeli hic fine-tune etmesek, sadece hazir embedding'lerini kullansak ne olur"
  sorusunun cevabidir.
- "full": tum model (BERT govdesi + siniflandirma katmani) birlikte egitilir.

Ikisi de ayni 114 ornek train setinde egitilir, ayni 51 ornek test setinde olculur —
tek degisken (egitim modu) disinda hicbir sey degismez.

Calistirma:
  PYTHONPATH=. python app/intent/train_classifier.py --mode frozen
  PYTHONPATH=. python app/intent/train_classifier.py --mode full
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from app.intent.rule_based_router import INTENTS

BASE_MODEL = "dbmdz/bert-base-turkish-cased"
LABEL2ID = {label: i for i, label in enumerate(INTENTS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def load_split(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return Dataset.from_dict({
        "text": [r["text"] for r in records],
        "label": [LABEL2ID[r["intent"]] for r in records],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["frozen", "full"], required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output or f"models/intent_{args.mode}"
    epochs = args.epochs or (15 if args.mode == "frozen" else 5)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(INTENTS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    if args.mode == "frozen":
        for param in model.bert.parameters():
            param.requires_grad = False

    train_ds = load_split("data/intent/train.json")
    test_ds = load_split("data/intent/test.json")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-5 if args.mode == "full" else 1e-3,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    print(f"=== Egitim basliyor: mode={args.mode}, epochs={epochs}, device={trainer.args.device} ===")
    train_result = trainer.train()

    # Test setinde tahmin olasiliklarini (softmax) al
    predictions = trainer.predict(test_ds)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_pred = np.argmax(logits, axis=-1)
    y_true = predictions.label_ids

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    results = {
        "mode": args.mode,
        "epochs": epochs,
        "train_loss": train_result.training_loss,
        "y_true": [ID2LABEL[i] for i in y_true.tolist()],
        "y_pred": [ID2LABEL[i] for i in y_pred.tolist()],
        "probs": probs.tolist(),
        "labels_order": INTENTS,
    }
    with open(f"{output_dir}/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    accuracy = (y_pred == y_true).mean()
    print(f"\n=== Sonuc (mode={args.mode}) ===")
    print(f"Train loss: {train_result.training_loss:.4f}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Model kaydedildi: {output_dir}")


if __name__ == "__main__":
    main()
