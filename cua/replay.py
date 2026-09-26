"""Deterministic replay: execute a Capability with typed inputs and no LLM in the loop.

Per step:
  1. ensure automation holds control of the session (handoff seam);
  2. check for known runtime states (app profile) before acting;
  3. irreversible step -> approval gate (human decides; never silently executed);
  4. resolve the target by (frame, role, accessible name, nth), waiting for it to appear -- fallbacks are
     tried only after the primary locator fails and are reported as drift;
  5. act;
  6. assert the step's postcondition (checkpoint). While waiting, every poll classifies the screen against the
     app profile's known states:
        business      -> stop, return a typed business outcome (not an error)
        recoverable   -> apply the bounded recovery (click / wait / re-run a flow + restart) and keep waiting
        hard_failure  -> stop with a debuggable failure
     A screen that matches neither the checkpoint nor any known state before the timeout is *unrecognized*:
     escalate to a human (if enabled) or fail with expected-vs-observed evidence.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any

from cua.evidence import RunLog
from cua.handoff import Controller
from cua.policy import Policy
from cua.schema import (
    AppProfile, BusinessOutcome, Capability, Checkpoint, Click, DriftRecord, Extract, Failure, Fill, KnownState,
    Navigate, OutputSpec, ParamValue, Press, RecoveryRecord, ReplayResult, Select, Sensitivity, StateView, Step,
    SecretValue, TenantConfig, ValueExpr,
)
from cua.surface.base import Observation, TargetError
from cua.surface.web import render_pattern

POLL_MS = 200
SETTLE_MS = 1500  # how long a screen must persist (without a navigation) before we classify it
FLOW_ENTRY_STATES = {"signed_out", "session_expired"}


class _Business(Exception):
    def __init__(self, outcome: BusinessOutcome):
        self.outcome = outcome


class _Fail(Exception):
    def __init__(self, failure: Failure):
        self.failure = failure


class _Restart(Exception):
    pass


def coerce_output(spec: OutputSpec, raw: str) -> Any:
    raw = raw.strip()
    if spec.type == "money":
        neg = raw.startswith("(") or raw.startswith("-")
        try:
            v = Decimal(re.sub(r"[^\d.]", "", raw))
        except InvalidOperation:
            raise ValueError(f"not a money amount: {raw!r}") from None
        return str((-v if neg else v).quantize(Decimal("0.01")))
    if spec.type == "integer":
        return int(re.sub(r"[^\d-]", "", raw))
    return raw


class Replayer:
    def __init__(self, surface, profile: AppProfile, tenant: TenantConfig, policy: Policy, log: RunLog,
                 controller: Controller, escalate: bool = True):
        self.s = surface
        self.profile = profile
        self.tenant = tenant
        self.policy = policy
        self.log = log
        self.ctl = controller
        self.escalate_enabled = escalate
        self.aliases = tenant.aliases
        self.params: dict[str, str] = {}
        self._attempts: Counter[str] = Counter()
        self._irreversible_done = False
        self._recoveries: list[RecoveryRecord] = []
        self._drift: list[DriftRecord] = []
        self._interventions: list[str] = []
        self._cap: Capability | None = None

    # ================================================================== public
    def run(self, cap: Capability, raw_params: dict[str, Any], allow_draft: bool = False) -> ReplayResult:
        t0 = time.monotonic()
        self._cap = cap
        outputs: dict[str, Any] = {}
        done = 0
        base = dict(capability=cap.ref, run_id=self.log.run_id, tenant_id=self.tenant.tenant_id)
        self.log.event("replay_started", capability=cap.ref, status=cap.status, tenant=self.tenant.tenant_id,
                       inputs=sorted(raw_params))
        try:
            self._validate_request(cap, raw_params, allow_draft)
            self.s.open(cap.entry_route)
            self._settle()
            if cap.requires_session:
                self._ensure_session()
            restarts = 0
            i = 0
            while i < len(cap.steps):
                step = cap.steps[i]
                try:
                    val = self._exec_step(step, in_flow=False)
                except _Restart:
                    restarts += 1
                    self.log.event("flow_restart", after_step=step.id, restarts=restarts)
                    self.s.open(cap.entry_route)
                    self._settle()
                    outputs.clear()
                    i = 0
                    continue
                if isinstance(step.action, Extract):
                    outputs[step.action.output] = val
                done = i + 1
                i += 1
            self._await(cap.success, None, self.s.navigation_count(), label="final success condition")
            status, extra = "succeeded", {"outputs": outputs}
            self.log.event("replay_succeeded", outputs=self._loggable_outputs(cap, outputs))
        except _Business as b:
            status, extra = "business_outcome", {"outcome": b.outcome}
            self.log.event("business_outcome", code=b.outcome.code, message=b.outcome.message, step=b.outcome.step_id)
        except _Fail as f:
            status, extra = "failed", {"failure": f.failure}
            self.log.event("replay_failed", **f.failure.model_dump(exclude_none=True))
        result = ReplayResult(status=status, **base, **extra, recoveries=self._recoveries, drift=self._drift,
                              interventions=self._interventions, steps_completed=done,
                              duration_ms=int((time.monotonic() - t0) * 1000))
        red = result.model_dump(mode="json")
        red["outputs"] = self._loggable_outputs(cap, result.outputs)
        self.log.write_json("result.json", red)
        return result

    # session helpers shared with discovery
    def settle(self) -> None:
        self._settle()

    def ensure_session(self) -> None:
        try:
            self._ensure_session()
        except _Fail as f:
            raise RuntimeError(f"sign-on failed: {f.failure.message}") from None

    # ================================================================== request validation
    def _validate_request(self, cap: Capability, raw: dict[str, Any], allow_draft: bool) -> None:
        def bad(msg: str, code: str = "invalid_request") -> _Fail:
            return _Fail(Failure(code=code, message=msg))

        if cap.status == "deprecated":
            raise bad(f"{cap.ref} is deprecated")
        if cap.status == "draft" and not allow_draft:
            raise bad(f"{cap.ref} is a draft; it must be reviewed and approved before unattended use")
        want = f"{cap.app.profile}@{cap.app.profile_version}"
        if self.tenant.app_profile != want or f"{self.profile.id}@{self.profile.version}" != want:
            raise bad(f"capability targets {want}, tenant runs {self.tenant.app_profile}")
        unknown = set(raw) - set(cap.inputs)
        if unknown:
            raise bad(f"unknown inputs: {sorted(unknown)}")
        for name, spec in cap.inputs.items():
            if name not in raw:
                if spec.required:
                    raise bad(f"missing required input '{name}'")
                continue
            try:
                self.params[name] = spec.validate_value(raw[name])
            except ValueError as e:
                raise bad(f"input '{name}' {e}") from None
            if spec.sensitivity in (Sensitivity.pii, Sensitivity.secret):
                self.log.redactor.add_pii(self.params[name])
        problems = self.policy.check_capability(cap)
        if problems:
            raise bad("; ".join(problems), "policy_violation")

    # ================================================================== steps
    def _exec_step(self, step: Step, in_flow: bool) -> Any:
        self.ctl.ensure_automation()
        self._check_known_states_now(step, in_flow)
        a = step.action
        self.log.event("step_started", step=step.id, intent=step.intent, action=a.type, risk=step.risk,
                       target=getattr(getattr(a, "target", None), "describe", lambda: None)())
        if step.risk == "irreversible":
            if self._approval_gate(step) == "done_by_human":
                return None
        nav_before = self.s.navigation_count()
        value = None
        if isinstance(a, Navigate):
            route = render_pattern(a.route, self.params)
            ok, why = self.policy.url_allowed(self.s.base_url + route)
            if not ok:
                raise _Fail(self._failure("policy_violation", why, step))
            self.s.navigate(route)
        elif isinstance(a, Extract):
            value = self._extract(step)
        else:
            r = self._resolve(step, a.target)
            if isinstance(a, Click):
                self.s.click(r)
            elif isinstance(a, Fill):
                self.s.fill(r, self._value(a.value))
            elif isinstance(a, Select):
                self.s.select(r, self._value(a.value))
            elif isinstance(a, Press):
                self.s.press(r, a.key)
        if step.risk == "irreversible":
            self._irreversible_done = True
        if self.s.violations:
            v = self.s.violations.pop(0)
            raise _Fail(self._failure("policy_violation", f"blocked navigation to {v['url']}: {v['reason']}", step))
        if step.expect:
            self._await(step.expect, step, nav_before, in_flow=in_flow)
        self.log.event("step_completed", step=step.id)
        return value

    def _value(self, v: ValueExpr) -> str:
        if isinstance(v, ParamValue):
            return self.params[v.name]
        if isinstance(v, SecretValue):
            env = self.tenant.secrets.get(v.name)
            val = os.environ.get(env or "")
            if not val:
                raise _Fail(Failure(code="invalid_request", message=f"secret '{v.name}' not configured for tenant {self.tenant.tenant_id}"))
            self.log.redactor.add_secret(val)
            return val
        return v.value

    def _resolve(self, step: Step, target, timeout_ms: int = 6000):
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            try:
                r = self.s.resolve(target, self.aliases)
                if r.strategy != "primary":
                    self._drift.append(DriftRecord(step_id=step.id, target=target.describe(), strategy=r.strategy))
                    self.log.event("locator_drift", step=step.id, target=target.describe(), strategy=r.strategy)
                return r
            except TargetError as e:
                if e.code == "target_ambiguous":
                    raise _Fail(self._failure("target_ambiguous", str(e), step, expected=target.describe()))
                last = e
            obs = self.s.observe(with_elements=False)
            ks = self._match_known(obs)
            if ks:
                self._handle_known(ks, step, obs, in_flow=False)
                continue
            if time.monotonic() > deadline:
                raise _Fail(self._failure("target_not_found", str(last), step, expected=target.describe()))
            self.s.pause(POLL_MS)

    def _extract(self, step: Step) -> Any:
        a: Extract = step.action  # type: ignore[assignment]
        spec = self._cap.outputs[a.output]
        row_key = None
        if getattr(a.source, "row_key", None) is not None:
            row_key = self._value(a.source.row_key)
        raw = None
        for _ in range(10):
            raw = self.s.extract(a.source, row_key, self.aliases)
            if raw:
                break
            self.s.pause(POLL_MS)
        if not raw:
            raise _Fail(self._failure("output_invalid", f"could not read output '{a.output}'", step, expected=a.source.kind))
        try:
            val = coerce_output(spec, raw)
        except ValueError as e:
            raise _Fail(self._failure("output_invalid", f"output '{a.output}': {e}", step)) from None
        if spec.sensitivity in (Sensitivity.pii, Sensitivity.secret):
            self.log.redactor.add_pii(str(val))
        self.log.event("output_extracted", step=step.id, output=a.output,
                       value=val if spec.sensitivity in (Sensitivity.public, Sensitivity.internal) else "[MASKED]")
        return val

    # ================================================================== checkpoints & known states
    def _satisfied(self, cp: Checkpoint, obs: Observation) -> bool:
        return all(self.s.check(c, self.params, self.aliases, obs) for c in cp.all_of)

    def _await(self, cp: Checkpoint, step: Step | None, nav_before: int, label: str = "postcondition", in_flow: bool = False) -> None:
        start = time.monotonic()
        deadline = start + cp.timeout_ms / 1000
        while True:
            obs = self.s.observe(with_elements=False)
            if self._satisfied(cp, obs):
                return
            # Only classify a screen that is the *result* of the action: a navigation has committed and nothing
            # is still loading, or the screen has persisted for SETTLE_MS (in-page changes without navigation).
            navigated = self.s.navigation_count() > nav_before
            loading = self.s.navigation_pending()
            if not loading and (navigated or (time.monotonic() - start) * 1000 > SETTLE_MS):
                ks = self._match_known(obs)
                if ks and in_flow and ks.code in FLOW_ENTRY_STATES:
                    ks = None  # the sign-on flow starts on these screens; only real failures count here
                if ks:
                    self._handle_known(ks, step, obs, in_flow)
                    nav_before = self.s.navigation_count()
                    start = time.monotonic()
                    deadline = max(deadline, start + cp.timeout_ms / 1000)
                    continue
            if time.monotonic() > deadline:
                expected = " AND ".join(c.model_dump_json(exclude_none=True) for c in cp.all_of)
                if self._escalate_unrecognized(step, f"{label} not reached and screen matches no known state", cp):
                    return
                raise _Fail(self._failure("checkpoint_timeout", f"{label} not met within {cp.timeout_ms}ms; screen matches no known state",
                                          step, expected=expected, obs=obs))
            self.s.pause(POLL_MS)

    def _match_known(self, obs: Observation) -> KnownState | None:
        for ks in self.profile.known_states:
            if any(self.s.check(c, self.params, self.aliases, obs) for c in ks.detect_any):
                return ks
        return None

    def _check_known_states_now(self, step: Step, in_flow: bool) -> None:
        obs = self.s.observe(with_elements=False)
        ks = self._match_known(obs)
        if ks and not (in_flow and ks.code in FLOW_ENTRY_STATES):
            self._handle_known(ks, step, obs, in_flow)

    def _message(self, ks: KnownState, obs: Observation) -> str | None:
        if not ks.message_regex:
            return None
        for f in obs.frames:
            m = re.search(ks.message_regex, f.text)
            if m:
                return m.group(1).strip()
        return None

    def _handle_known(self, ks: KnownState, step: Step | None, obs: Observation, in_flow: bool) -> None:
        sid = step.id if step else None
        self.log.event("known_state_detected", code=ks.code, classification=ks.classification, step=sid)
        if ks.classification == "business":
            raise _Business(BusinessOutcome(code=ks.code, message=self.log.redactor.text(self._message(ks, obs) or ""), step_id=sid))
        if ks.classification == "hard_failure":
            raise _Fail(self._failure("host_error", f"{ks.code}: {self._message(ks, obs) or ks.description}", step, obs=obs))
        rec = ks.recovery
        self._attempts[ks.code] += 1
        if rec is None or self._attempts[ks.code] > rec.max_attempts:
            raise _Fail(self._failure("recovery_exhausted", f"{ks.code} persisted after {rec.max_attempts if rec else 0} recovery attempt(s)", step, obs=obs))
        if in_flow and rec.kind == "run_flow":
            raise _Fail(self._failure("recovery_exhausted", f"{ks.code} while already recovering", step, obs=obs))
        self._recoveries.append(RecoveryRecord(code=ks.code, step_id=sid, action=f"{rec.kind}:{rec.flow or (rec.target.describe() if rec.target else rec.wait_ms)}"))
        self.log.event("recovery", code=ks.code, action=rec.kind, attempt=self._attempts[ks.code])
        if rec.kind == "wait":
            self.s.pause(rec.wait_ms)
        elif rec.kind == "click":
            nav = self.s.navigation_count()
            self.s.click(self.s.resolve(rec.target, self.aliases))
            self._wait_nav(nav)
        elif rec.kind == "run_flow":
            if self._irreversible_done:
                raise _Fail(self._failure("recovery_exhausted", f"{ks.code} after an irreversible step; restarting could duplicate the transaction", step, obs=obs))
            self._run_flow(rec.flow)
            if rec.then == "restart":
                raise _Restart()

    def _wait_nav(self, before: int, ms: int = 5000) -> None:
        end = time.monotonic() + ms / 1000
        while self.s.navigation_count() == before and time.monotonic() < end:
            self.s.pause(100)
        self.s.pause(150)

    def _run_flow(self, name: str | None) -> None:
        flow = self.profile.flows.get(name or "")
        if flow is None:
            raise _Fail(Failure(code="surface_error", message=f"app profile has no flow '{name}'"))
        self.log.event("flow_started", flow=name)
        for st in flow.steps:
            self._exec_step(st, in_flow=True)
        self._await(flow.success, None, self.s.navigation_count(), label=f"flow '{name}' success", in_flow=True)
        self.log.event("flow_completed", flow=name)

    def _ensure_session(self) -> None:
        obs = self.s.observe(with_elements=False)
        if any(self.s.check(c, self.params, self.aliases, obs) for c in self.profile.signed_out_detect):
            self._run_flow(self.profile.sign_on_flow)

    def _settle(self) -> None:
        end = time.monotonic() + 10
        while time.monotonic() < end:
            obs = self.s.observe(with_elements=False)
            if all(f.title for f in obs.frames if f.name):
                return
            self.s.pause(POLL_MS)

    # ================================================================== humans
    def _approval_gate(self, step: Step) -> str:
        cap = self._cap
        if not self.policy.needs_approval(cap.id, cap.status):
            self.log.event("approval_waived", step=step.id, reason="capability approved for unattended irreversible use")
            return "approved"
        if self.policy.irreversible_mode == "block":
            raise _Fail(self._failure("policy_violation", "irreversible steps are blocked by policy", step))
        req = self.ctl.escalate(
            "approval", f"Step '{step.intent}' is irreversible. Approve to let automation perform it, deny to stop, "
            "or take control, perform it yourself, and resume.", cap.ref, allowed=["approve", "deny", "resume"],
            step_id=step.id, step_intent=step.intent, proposed_action=f"{step.action.type} {step.action.target.describe()}")
        self._interventions.append(req.id)
        if req.decision == "approve":
            return "approved"
        if req.decision == "resume":
            if step.expect and self._satisfied(step.expect, self.s.observe(with_elements=False)):
                self._irreversible_done = True
                return "done_by_human"
            raise _Fail(self._failure("unrecognized_state", "operator resumed but the step's postcondition does not hold", step))
        if req.decision == "deny":
            raise _Fail(self._failure("approval_denied", f"operator {req.resolved_by} denied: {req.note or 'no reason'}", step))
        raise _Fail(self._failure("escalation_aborted", f"approval request {req.decision}", step))

    def _escalate_unrecognized(self, step: Step | None, why: str, cp: Checkpoint) -> bool:
        """Returns True if a human fixed things and the checkpoint now holds."""
        if not self.escalate_enabled:
            return False
        req = self.ctl.escalate("stuck", why, self._cap.ref, allowed=["resume", "abort"],
                                step_id=step.id if step else None, step_intent=step.intent if step else None)
        self._interventions.append(req.id)
        if req.decision == "resume" and self._satisfied(cp, self.s.observe(with_elements=False)):
            self.log.event("resumed_after_handoff", step=step.id if step else None)
            return True
        return False

    # ================================================================== evidence
    def _failure(self, code: str, message: str, step: Step | None, expected: str | None = None,
                 obs: Observation | None = None) -> Failure:
        obs = obs or self.s.observe(with_elements=True)
        n = len(list((self.log.dir / "screenshots").glob("*.png"))) + 1
        shot = f"screenshots/failure-{n}.png"
        dom = f"dom/failure-{n}.txt"
        self.s.screenshot(str(self.log.dir / shot), self.profile.pii_labels)
        if not obs.elements:
            obs = self.s.observe(with_elements=True)
        (self.log.dir / dom).write_text(self.log.redactor.text(self.s.dom_digest(obs)), encoding="utf-8")
        view = StateView(frames=self.log.redactor.obj(
            [{"frame": f.name, "path": f.path, "title": f.title, "text": f.text[:400]} for f in obs.frames if f.name or f.title]))
        return Failure(code=code, message=self.log.redactor.text(message), step_id=step.id if step else None,
                       step_intent=step.intent if step else None, expected=expected, observed=view,
                       evidence={"screenshot": shot, "dom": dom})

    @staticmethod
    def _loggable_outputs(cap: Capability, outputs: dict[str, Any]) -> dict[str, Any]:
        return {k: (v if cap.outputs[k].sensitivity in (Sensitivity.public, Sensitivity.internal) else "[MASKED]")
                for k, v in outputs.items()}
