# P1 — AI Passenger Experience Assistant — Adım 10.
#
# Build context proje kökü olmalı (bkz. docker-compose.yml'deki `context: ..`) çünkü
# requirements.txt, app/, data/, models/ hepsi kökte — Dockerfile'ı docker/ altında
# tutup context'i kökten vermek, tek bir .dockerignore ile hem yerel çalıştırmayı hem
# build'i aynı dosya ağacından yönetmemizi sağlıyor.
#
# `models/intent_full_10ep/` imaja BİLİNÇLİ olarak dahil ediliyor (.dockerignore'a
# bakın) — bu ~420MB'lık ağırlık git'e girmiyor (GitHub'ın 100MB dosya sınırını aşıyor)
# ama planning_agent.py çalışma zamanında ona ihtiyaç duyuyor; imaj boyutu ile git
# deposu boyutu bilinçli olarak AYRI kararlar.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/ data/
COPY models/intent_full_10ep/ models/intent_full_10ep/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
