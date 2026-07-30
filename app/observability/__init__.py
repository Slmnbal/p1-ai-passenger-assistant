"""Observability & audit log katmanı.

Doldurulacağı adım: Adım 9
- structured_logging.py: JSON log + correlation id
- tracing.py: Langfuse (self-hosted) ile LLM/agent trace, token ve latency ölçümü
- metrics.py: Prometheus client ile sistem metrikleri (Grafana dashboard'u Adım 9'da)
- audit_log.py: kim, ne zaman, hangi işlemi yaptı kaydı

Not: LangSmith değil Langfuse kullanılıyor çünkü LangSmith belirli kullanımdan sonra
ücretli katmana geçiyor; "para harcanmayacak" kısıtıyla çelişir (bkz. p1_proje_plani.md).
"""
