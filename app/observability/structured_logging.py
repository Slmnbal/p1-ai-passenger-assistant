"""JSON yapılandırılmış loglama + correlation id — Adım 9.

Neden JSON (düz metin değil): "Bir isteğin baştan sona izlenebilmesi" (plan, Adım 9
çıktısı) için loglar makine tarafından ayrıştırılabilir olmalı — `request_id` alanına
göre filtrelenip tek bir isteğin tüm node'lardaki log satırları bir arada görülebilmeli
(örn. `grep '"request_id": "abc123"' | jq`).

`request_id`, `graph.py`'de her `run()` çağrısında üretilip `ConversationState`'e
yazılıyor (bkz. state.py) — Langfuse trace id'siyle AYNI değer kullanılıyor, böylece
bir isteğin hem JSON loglarını hem Langfuse trace'ini aynı id ile eşleştirmek mümkün.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

_LOGGER_NAME = "p1"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            payload["request_id"] = request_id
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(level or os.environ.get("LOG_LEVEL", "INFO"))
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    request_id: str | None = None,
    **fields,
) -> None:
    logger.log(level, message, extra={"request_id": request_id, "extra_fields": fields})
