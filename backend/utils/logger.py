"""Structured JSON logging for DataMind.

Never logs secrets/API keys. Each log line is a single JSON object so it can be
ingested by any log aggregator. Use `get_logger(__name__)` in every module.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_REDACT_KEYS = {"api_key", "apikey", "token", "password", "secret", "authorization"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            for key, value in extra.items():
                if key.lower() in _REDACT_KEYS:
                    value = "***redacted***"
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ExtraAdapter(logging.LoggerAdapter):
    """Allows logger.info("msg", extra_fields={...}) style structured logging."""

    def process(self, msg, kwargs):
        return msg, kwargs


_configured = False


def _configure_root(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("datamind")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.LoggerAdapter:
    try:
        from backend.config import settings

        _configure_root(settings.log_level)
    except Exception:
        _configure_root("INFO")
    base_logger = logging.getLogger(f"datamind.{name}")
    return ExtraAdapter(base_logger, {})
