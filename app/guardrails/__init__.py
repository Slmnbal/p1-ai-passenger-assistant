"""Guardrails & output verification katmanı (Adım 7, tamamlandı).

- schemas.py: `GuardrailReport` + tool_result üzerinde iş kuralı kontrolleri
  (requires_human_approval gibi sabit iş kurallarının bozulmadığını doğrular)
- prompt_injection_tests.py: iki katmanlı tespit — kural/keyword + embedding-tabanlı
  anlamsal benzerlik (`detect_injection_semantic`). İkincisi parafraze saldırıları
  (örn. "kısıtlaması olmayan asistan" cümlesi) yakalıyor ama roleplay/hikaye-çerçeveli
  dolaylı saldırılara karşı hâlâ savunmasız, bkz. docs/adr/0003 Deney 5.
- grounding.py: zorunlu kaynak atfı (`require_sources`) + claim decomposition ile
  faithfulness judge (`judge_faithfulness` — cevabı iddialara bölüp her birini ayrı
  doğruluyor, %90 doğru ölçüldü, bkz. docs/adr/0003 Deney 4). Artık BLOKLAYICI.
- numeric_template.py: cevaptaki sayıların (kg/cm/TRY/%) retrieved chunk'larla
  tutarlılığını deterministik (regex) kontrol eder
- confidence_fallback.py: retrieval-skor-eşiğinin neden KULLANILMADIĞI (ölçüldü, işe
  yaramadı) ve intent güveninin neden sert blokaj değil gözlem bayrağı olduğu

`app/agent/graph.py`'deki `input_guard_node` (planning'den önce) ve `output_guard_node`
(tools/verification'dan sonra) bu modülleri çağırır. Tüm kararların ölçüm/deney detayı
için bkz. `docs/adr/0003-guardrail-katmani-guven-esigi-ve-llm-judge-olcumu.md`.
"""
