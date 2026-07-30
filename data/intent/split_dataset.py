"""165 ornegi sinif oranlarini koruyarak train/test olarak boler (stratified split).

Sabit random seed ile tekrarlanabilir. Test orani %30 (kucuk veri setinde daha stabil
metrik icin standart %20'den biraz daha yuksek tutuldu).

Calistirma: python data/intent/split_dataset.py
Cikti: data/intent/train.json, data/intent/test.json
"""

from __future__ import annotations

import json
import random

random.seed(42)
TEST_FRACTION = 0.3


def main() -> None:
    with open("data/intent/messages.json", encoding="utf-8") as f:
        records = json.load(f)

    by_intent: dict[str, list] = {}
    for r in records:
        by_intent.setdefault(r["intent"], []).append(r)

    train, test = [], []
    for intent, items in by_intent.items():
        shuffled = items[:]
        random.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * TEST_FRACTION))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    with open("data/intent/train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open("data/intent/test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train)}, Test: {len(test)}")
    for intent in sorted(by_intent):
        n_tr = sum(1 for r in train if r["intent"] == intent)
        n_te = sum(1 for r in test if r["intent"] == intent)
        print(f"  {intent}: train={n_tr}, test={n_te}")


if __name__ == "__main__":
    main()
