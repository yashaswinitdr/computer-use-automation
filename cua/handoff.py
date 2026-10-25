"""Human-in-the-loop escalation and live-session control transfer.

Control model
-------------
Each run has exactly one *controller* of its live browser session at any time, recorded in
``runs/<run_id>/handoff/control.json`` as ``{holder: automation|human, epoch, operator}``. The epoch increments on
every transfer, so a stale actor can always tell it has lost control.

* Automation calls ``ensure_automation()`` before every UI action. If a human holds control it will not act.
* When automation cannot safely proceed it *escalates*: it writes an ``InterventionRequest`` (why, which step,
  redacted state, masked screenshot, allowed decisions), then blocks -- keeping the live session open.
* An operator *claims* the request (control -> human, epoch+1), works in the same live browser session
  (run with --headed; a production console would stream it, see REPORT.md), and then *resolves* it with a
  decision (control -> automation, epoch+1).
* Every DOM-level action the human performs while holding control is captured by the same perception library
  used for automation (role + accessible name + frame), logged, and attached to the request.
* On resume the automation re-observes the surface and re-verifies state before doing anything -- it never
  assumes the human left the UI where it was.

The store is plain JSON files so the operator side (console, CLI, or a real ticketing/queue integration) can run
in a different process from the automation.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

Decision = Literal["approve", "deny", "resume", "abort", "timeout"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ControlState(BaseModel):
    run_id: str
    holder: Literal["automation", "human"] = "automation"
    epoch: int = 0
    operator: str | None = None
    since: str = Field(default_factory=_now)
    reason: str = "run started"


class InterventionRequest(BaseModel):
    id: str
    run_id: str
    kind: Literal["stuck", "approval", "unrecoverable"]
    reason: str
    subject: str  # capability ref or discovery goal (redacted)
    step_id: str | None = None
    step_intent: str | None = None
    proposed_action: str | None = None
    state: list[dict[str, Any]] = []
    screenshot: str | None = None
    allowed_decisions: list[Decision]
    status: Literal["open", "in_progress", "resolved"] = "open"
    created_at: str = Field(default_factory=_now)
    claimed_by: str | None = None
    claimed_at: str | None = None
    decision: Decision | None = None
    note: str | None = None
    resolved_by: str | None = None
    resolved_at: str | None = None
    human_actions: list[dict[str, Any]] = []


class HandoffStore:
    def __init__(self, runs_root: Path):
        self.root = Path(runs_root)

    def _dir(self, run_id: str) -> Path:
        d = self.root / run_id / "handoff"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _write(path: Path, model: BaseModel) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    # -- control
    def control(self, run_id: str) -> ControlState:
        p = self._dir(run_id) / "control.json"
        if not p.exists():
            st = ControlState(run_id=run_id)
            self._write(p, st)
            return st
        return ControlState.model_validate_json(p.read_text(encoding="utf-8"))

    def _transfer(self, run_id: str, holder: str, operator: str | None, reason: str) -> ControlState:
        cur = self.control(run_id)
        st = ControlState(run_id=run_id, holder=holder, epoch=cur.epoch + 1, operator=operator, reason=reason)
        self._write(self._dir(run_id) / "control.json", st)
        with open(self._dir(run_id) / "control_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(st.model_dump_json() + "\n")
        return st

    # -- requests
    def open_request(self, req: InterventionRequest) -> None:
        self._write(self._dir(req.run_id) / f"{req.id}.json", req)

    def load(self, run_id: str, req_id: str) -> InterventionRequest:
        return InterventionRequest.model_validate_json((self._dir(run_id) / f"{req_id}.json").read_text(encoding="utf-8"))

    def save(self, req: InterventionRequest) -> None:
        self._write(self._dir(req.run_id) / f"{req.id}.json", req)

    def list(self, status: str | None = "open") -> list[InterventionRequest]:
        out = []
        for p in sorted(self.root.glob("*/handoff/req-*.json")):
            r = InterventionRequest.model_validate_json(p.read_text(encoding="utf-8"))
            if status is None or r.status == status or (status == "open" and r.status == "in_progress"):
                out.append(r)
        return out

    def claim(self, run_id: str, req_id: str, operator: str) -> InterventionRequest:
        req = self.load(run_id, req_id)
        if req.status != "open":
            raise ValueError(f"request {req_id} is {req.status}")
        req.status, req.claimed_by, req.claimed_at = "in_progress", operator, _now()
        self.save(req)
        self._transfer(run_id, "human", operator, f"claimed {req_id}")
        return req

    def resolve(self, run_id: str, req_id: str, decision: Decision, operator: str, note: str | None = None) -> InterventionRequest:
        req = self.load(run_id, req_id)
        if req.status == "resolved":
            raise ValueError(f"request {req_id} already resolved")
        if decision not in req.allowed_decisions and decision not in ("abort", "timeout"):
            raise ValueError(f"decision {decision} not allowed for this request ({req.allowed_decisions})")
        req.status, req.decision, req.note, req.resolved_by, req.resolved_at = "resolved", decision, note, operator, _now()
        self.save(req)
        if self.control(run_id).holder != "automation":
            self._transfer(run_id, "automation", None, f"resolved {req_id}: {decision}")
        return req


class OperatorHook(Protocol):
    """Called while automation is blocked on a request. A real deployment's hook is a no-op (humans act through the
    console / live browser); ``SimulatedOperator`` uses it to stand in for a person in demos and tests."""

    def tick(self, req: InterventionRequest, store: HandoffStore, surface: Any) -> None: ...


class ControlLost(Exception):
    pass


class Controller:
    """Automation-side half of the handoff seam."""

    def __init__(self, store: HandoffStore, run_id: str, log, surface, operator: OperatorHook | None = None,
                 timeout_s: float = 900, pii_labels: list[str] | None = None):
        self.store, self.run_id, self.log, self.surface = store, run_id, log, surface
        self.operator = operator
        self.timeout_s = timeout_s
        self.pii_labels = pii_labels or []
        self.enabled = True
        self.last_raw_actions: list[dict[str, Any]] = []  # unredacted, in-memory only (used by the recorder)
        store.control(run_id)  # initialise control.json

    def ensure_automation(self) -> None:
        st = self.store.control(self.run_id)
        if st.holder != "automation":
            raise ControlLost(f"control held by {st.holder} ({st.operator}) at epoch {st.epoch}")

    def escalate(self, kind: str, reason: str, subject: str, allowed: list[Decision], step_id: str | None = None,
                 step_intent: str | None = None, proposed_action: str | None = None) -> InterventionRequest:
        req_id = f"req-{secrets.token_hex(3)}"
        shot = f"screenshots/{req_id}.png"
        self.surface.screenshot(str(self.log.dir / shot), self.pii_labels)
        obs = self.surface.observe(with_elements=True)
        (self.log.dir / "dom" / f"{req_id}.txt").write_text(self.log.redactor.text(self.surface.dom_digest(obs)), encoding="utf-8")
        state = [{"frame": f.name, "path": f.path, "title": f.title, "text": f.text[:600]} for f in obs.frames]
        req = InterventionRequest(
            id=req_id, run_id=self.run_id, kind=kind, reason=reason, subject=subject, step_id=step_id,
            step_intent=step_intent, proposed_action=proposed_action, state=self.log.redactor.obj(state),
            screenshot=shot, allowed_decisions=allowed,
        )
        req.subject = self.log.redactor.text(req.subject)
        req.reason = self.log.redactor.text(req.reason)
        self.store.open_request(req)
        self.log.event("intervention_requested", request=req_id, kind=kind, reason=reason, step=step_id,
                       allowed=allowed, screenshot=shot)
        if not self.enabled:
            return self.store.resolve(self.run_id, req_id, "abort", "system", "escalation disabled for this run")

        captured: list[dict[str, Any]] = []
        self.last_raw_actions = []

        def on_human(evt: dict[str, Any]) -> None:
            if self.store.control(self.run_id).holder != "human":
                return  # automation's own synthetic events are not human actions
            evt["action"] = evt.pop("type", None)
            if evt.get("secret"):
                evt["value"] = None
            self.last_raw_actions.append(dict(evt))
            evt = self.log.redactor.obj(evt)
            evt["at"] = _now()
            captured.append(evt)
            self.log.event("human_action", request=req_id, **evt)

        self.surface.on_human_action(on_human)
        deadline = time.monotonic() + self.timeout_s
        try:
            while True:
                if self.operator:
                    self.operator.tick(self.store.load(self.run_id, req_id), self.store, self.surface)
                self.surface.pause(250)
                cur = self.store.load(self.run_id, req_id)
                if cur.status == "resolved":
                    break
                if time.monotonic() > deadline:
                    cur = self.store.resolve(self.run_id, req_id, "timeout", "system", "no operator response")
                    break
        finally:
            self.surface.on_human_action(None)
        cur.human_actions = captured
        self.store.save(cur)
        log_path = self.store.root / self.run_id / "handoff" / "control_log.jsonl"
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                t = json.loads(line)
                if req_id in t.get("reason", ""):
                    self.log.event("control_transferred", request=req_id, holder=t["holder"], operator=t["operator"],
                                   epoch=t["epoch"], at=t["since"])
        self.log.event("intervention_resolved", request=req_id, decision=cur.decision, by=cur.resolved_by,
                       note=cur.note, human_actions=len(captured))
        return cur


class SimulatedOperator:
    """Stand-in for a human operator in demos/tests. It acts on the *live* page through real input events
    (Playwright mouse/keyboard), so the human-action recorder captures it exactly like a person.

    script: list of {"kind": <request kind>, "claim": bool, "actions": [...], "decision": ..., "note": ...}
    consumed in order, one per intervention request.
    """

    def __init__(self, script: list[dict[str, Any]], name: str = "sim-operator"):
        self.script = list(script)
        self.name = name
        self._handled: set[str] = set()

    def tick(self, req: InterventionRequest, store: HandoffStore, surface: Any) -> None:
        if req.id in self._handled or req.status == "resolved":
            return
        self._handled.add(req.id)
        plan = next((p for p in self.script if p.get("kind", req.kind) == req.kind), None)
        if plan is None:  # nothing scripted for this kind of request: a real operator would abort it
            store.resolve(req.run_id, req.id, "abort", self.name, f"no scripted response for a '{req.kind}' request")
            return
        self.script.remove(plan)
        if plan.get("claim", True):
            store.claim(req.run_id, req.id, self.name)
            surface.pause(300)
        for a in plan.get("actions", []):
            frame = next(f for f in surface.page.frames if f.name == a.get("frame", "main"))
            if "css" in a:
                loc = frame.locator(a["css"])
            else:  # find the control the way a person would: by its visible caption
                from cua.schema import Target

                loc = surface._loc(surface.resolve(Target(frame=a.get("frame", "main"), role=a["role"], name=a["name"]), {}))
            if a["type"] == "fill":
                loc.click()
                loc.press_sequentially(a["value"], delay=20)
                loc.press("Tab")
            elif a["type"] == "select":
                loc.select_option(label=a["value"])
            else:
                loc.click()
            surface.pause(a.get("pause_ms", 600))
        store.resolve(req.run_id, req.id, plan.get("decision", "resume"), self.name, plan.get("note"))


def load_operator_script(path: str | None) -> SimulatedOperator | None:
    if not path:
        return None
    return SimulatedOperator(json.loads(Path(path).read_text(encoding="utf-8")))
