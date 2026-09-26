"""Run directory + structured, redacted event log.

runs/<run_id>/
  events.jsonl       one JSON object per event (what happened and why), always redacted
  result.json        final result (redacted)
  screenshots/*.png  PII-masked screenshots (per step in discovery, on failure/escalation in replay)
  dom/*.txt          redacted text + control inventory of every frame, captured on failure/escalation
  handoff/*.json     intervention requests and control-transfer records
"""

from __future__ import annotations

import json
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cua.redact import Redactor


def new_run_id(kind: str) -> str:
    return f"{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class RunLog:
    def __init__(self, root: Path, run_id: str, redactor: Redactor, echo: bool = True):
        self.run_id = run_id
        self.dir = root / run_id
        (self.dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (self.dir / "dom").mkdir(exist_ok=True)
        (self.dir / "handoff").mkdir(exist_ok=True)
        self.redactor = redactor
        self.echo = echo
        self._t0 = time.monotonic()
        self._seq = 0
        self._fh = open(self.dir / "events.jsonl", "a", encoding="utf-8")

    def event(self, type_: str, **fields: Any) -> None:
        self._seq += 1
        rec = {
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "t_ms": int((time.monotonic() - self._t0) * 1000),
            "type": type_,
            **self.redactor.obj(fields),
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        if self.echo:
            summary = {k: v for k, v in rec.items() if k not in ("seq", "ts", "observation", "state")}
            line = json.dumps(summary, default=str)
            print(f"  [{rec['t_ms']:>6}ms] {line[:220]}", file=sys.stderr)

    def artifact_path(self, kind: str, name: str) -> Path:
        return self.dir / kind / name

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(self.redactor.obj(obj), indent=2, default=str), encoding="utf-8")
        return p

    def close(self) -> None:
        self._fh.close()
