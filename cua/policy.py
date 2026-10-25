"""Safety policy: what the automation may touch and how risky actions are treated.

Enforcement happens at three layers so that no single bug (or model output) can bypass it:
1. static check of an artifact before replay starts (every action type / route allowed);
2. per-action check in the agent loop and replay engine (action type, risk class -> approval gate);
3. network layer: every document/frame navigation is intercepted by the browser surface and aborted if its
   URL is outside the allowlist -- this catches navigations caused *indirectly* by a click.
"""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from cua.schema import Capability, Risk


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_origins: list[str]
    allowed_routes: list[str]  # glob patterns on the path
    denied_routes: list[str] = []
    allowed_actions: list[str] = ["click", "fill", "select", "press", "navigate", "extract"]
    # Controls whose accessible name matches one of these are classed irreversible (commit a transaction).
    irreversible_names: list[str] = [r"(?i)^(confirm|submit|post|transfer|delete|close account|approve)\b"]
    # Controls that write but can be undone / never leave a draft state.
    write_names: list[str] = [r"(?i)^(continue|save|next|update)\b"]
    # Controls that must never be activated by automation (e.g. ending the shared session).
    forbidden_names: list[str] = [r"(?i)^(sign off|log ?out)\b"]
    irreversible_mode: str = "require_approval"  # or "block"
    # Capability ids allowed to run irreversible steps unattended (only when status == approved).
    unattended_irreversible: list[str] = []

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        return cls.model_validate(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------ checks
    def url_allowed(self, url: str) -> tuple[bool, str]:
        parts = urlsplit(url)
        if parts.scheme in ("about", "data", "blob"):
            return True, ""
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.allowed_origins:
            return False, f"origin {origin} not in allowlist"
        path = parts.path or "/"
        if any(fnmatch.fnmatch(path, p) for p in self.denied_routes):
            return False, f"route {path} explicitly denied"
        if not any(fnmatch.fnmatch(path, p) for p in self.allowed_routes):
            return False, f"route {path} not in allowlist"
        return True, ""

    def action_allowed(self, action_type: str) -> tuple[bool, str]:
        if action_type not in self.allowed_actions:
            return False, f"action type '{action_type}' not allowed"
        return True, ""

    def classify(self, action_type: str, control_name: str | None) -> Risk | None:
        """Risk class for an action; None means forbidden outright."""
        name = control_name or ""
        if action_type in ("click", "press") and any(re.search(p, name) for p in self.forbidden_names):
            return None
        if action_type in ("click", "press"):
            if any(re.search(p, name) for p in self.irreversible_names):
                return "irreversible"
            if any(re.search(p, name) for p in self.write_names):
                return "write"
            return "read"
        if action_type in ("fill", "select"):
            return "read"  # local form state only: nothing reaches the system of record until a submit click
        return "read"

    def check_capability(self, cap: Capability) -> list[str]:
        problems = []
        for s in cap.steps:
            ok, why = self.action_allowed(s.action.type)
            if not ok:
                problems.append(f"{s.id}: {why}")
            name = getattr(getattr(s.action, "target", None), "name", None)
            risk = self.classify(s.action.type, name)
            if risk is None:
                problems.append(f"{s.id}: control '{name}' is forbidden")
            elif risk == "irreversible" and s.risk != "irreversible":
                problems.append(f"{s.id}: policy classes '{name}' as irreversible but artifact says {s.risk}")
        return problems

    def needs_approval(self, cap_id: str | None, cap_status: str | None) -> bool:
        if self.irreversible_mode == "block":
            return True
        return not (cap_status == "approved" and cap_id in self.unattended_irreversible)
