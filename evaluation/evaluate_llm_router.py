"""LLM (Ollama/llama3.1) ile dogrudan intent routing'i test setinde olcer.

Calistirma: PYTHONPATH=. python evaluation/evaluate_llm_router.py
"""

from __future__ import annotations

import json
import time

from app.intent.llm_router import build_client, classify_with_llm

MODEL = "llama3.1:8b"


def main() -> None:
    with open("data/intent/test.json", encoding="utf-8") as f:
        test_records = json.load(f)

    client = build_client()

    results = []
    start = time.time()
    for i, r in enumerate(test_records):
        pred = classify_with_llm(r["text"], client, MODEL)
        results.append({"text": r["text"], "y_true": r["intent"], "y_pred": pred})
        print(f"  [{i+1}/{len(test_records)}] {r['intent']:<30} -> {pred}")

    elapsed = time.time() - start
    print(f"\nToplam sure: {elapsed:.1f}s ({elapsed/len(test_records):.2f}s/ornek)")

    correct = sum(1 for r in results if r["y_true"] == r["y_pred"])
    print(f"Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")

    with open("evaluation/intent_llm_routing_results.json", "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "results": results, "elapsed_seconds": elapsed}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
