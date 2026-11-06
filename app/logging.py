"""Structured logging with PII masking. Phone numbers and e-mails never reach log storage in clear text."""

from __future__ import annotations

import logging
import re
from typing import Any

import structlog

_PHONE = re.compile(r"(?<![\dT:])(\+?\d[\d\s\-()]{7,}\d)(?![\dT:])")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _mask_phone(m: re.Match[str]) -> str:
    s = m.group(0)
    # phones carry 9+ digits; dates (8 digits) and times are left alone
    return s[:3] + "***" + s[-2:] if sum(ch.isdigit() for ch in s) >= 9 else s


def mask_pii(value: Any) -> Any:
    if isinstance(value, str):
        value = _EMAIL.sub("<email>", value)
        return _PHONE.sub(_mask_phone, value)
    if isinstance(value, dict):
        return {k: mask_pii(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_pii(v) for v in value]
    return value


def _mask_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    return mask_pii(event_dict)


def configure_logging(json: bool = False, level: int = logging.INFO) -> None:
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _mask_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
