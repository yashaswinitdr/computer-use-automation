"""Redaction applied to everything that leaves the process boundary: logs, evidence files, LLM prompts.

Two mechanisms:
* value-based: exact values we *know* are sensitive (resolved secrets, pii-classified params/outputs);
* pattern-based: shapes of regulated identifiers (SSN, card numbers, emails, phone numbers).

Limits (documented in REPORT.md): free-text PII such as a person's name is only caught when it is a known value
or sits next to a caption listed in the app profile's ``pii_labels``.
"""

from __future__ import annotations

import re
from typing import Any

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[A-Za-z][\w-]*(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\(?\b\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b")),
]


def _luhn(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def mask_value(v: str) -> str:
    """Keep a short suffix so humans can correlate runs without seeing the value."""
    return "***" if len(v) <= 4 else f"***{v[-2:]}"


class Redactor:
    def __init__(self) -> None:
        self._secrets: set[str] = set()
        self._pii: set[str] = set()

    def add_secret(self, value: str | None) -> None:
        if value:
            self._secrets.add(value)

    def add_pii(self, value: str | None) -> None:
        if value and len(value) >= 3:
            self._pii.add(value)

    def text(self, s: str) -> str:
        if not s:
            return s
        for v in sorted(self._secrets, key=len, reverse=True):
            s = s.replace(v, "[SECRET]")
        for v in sorted(self._pii, key=len, reverse=True):
            s = s.replace(v, mask_value(v))
        for label, pat in PATTERNS:
            if label == "CARD":
                s = pat.sub(lambda m: "[CARD]" if _luhn(m.group(0)) else m.group(0), s)
            else:
                s = pat.sub(f"[{label}]", s)
        return s

    def obj(self, o: Any) -> Any:
        if isinstance(o, str):
            return self.text(o)
        if isinstance(o, dict):
            return {k: ("[SECRET]" if k in ("password", "secret") else self.obj(v)) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.obj(v) for v in o]
        return o
