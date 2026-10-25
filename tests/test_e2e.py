"""End-to-end: scripted discovery -> artifact -> deterministic replay against the live (in-process) legacy app,
through a real Chromium. Covers the result contract (success / business outcome / failure), recoveries,
approval gating, human handoff and cross-tenant reuse. The LLM is replaced by ScriptedLLM here; the real
LLM discovery run is in evidence/.
"""

from __future__ import annotations

import json

import pytest

from cua.agent.discovery import Discovery
from cua.agent.llm import ScriptedLLM
from cua.handoff import SimulatedOperator
from cua.recorder import build_capability
from cua.replay import Replayer
from tests.conftest import ROOT

GOAL = "Look up member {member_id} and read their current share savings balance"


def script(name):
    return json.loads((ROOT / "scripts" / "scripted" / f"{name}.json").read_text())


@pytest.fixture
def savings_cap(env, faults):
    log, surface, ctl = env.open("discover")
    session = Replayer(surface, env.profile, env.tenant, env.policy, log, ctl)
    params = {"member_id": "10042"}
    out = Discovery(surface, ScriptedLLM(script("savings_balance")), env.policy, log, ctl, session,
                    env.profile.pii_labels).run(GOAL.format(**params), params)
    assert out.status == "succeeded", out.reason
    cap = build_capability(out, capability_id="t.savings", title="t", description="d", goal_template=GOAL,
                           params=params, profile=env.profile, run_id=log.run_id, model="scripted")
    return cap.model_copy(update={"status": "approved"})


def replay(env, cap, params, operator=None, escalate=False, tenant=None):
    log, surface, ctl = env.open("replay", operator=operator, escalate=escalate, tenant=tenant)
    t = tenant or env.tenant
    return Replayer(surface, env.profile, t, env.policy, log, ctl, escalate=escalate).run(cap, params), log


def test_discovery_logs_mask_pii_inputs(env, faults):
    log, surface, ctl = env.open("discover")
    session = Replayer(surface, env.profile, env.tenant, env.policy, log, ctl)
    params = {"member_id": "10042"}
    out = Discovery(surface, ScriptedLLM(script("savings_balance")), env.policy, log, ctl, session,
                    env.profile.pii_labels).run(GOAL.format(**params), params)
    assert out.status == "succeeded"
    log.write_json("trace.json", [t.__dict__ for t in out.trace])
    for f in ("events.jsonl", "trace.json"):
        assert "10042" not in (log.dir / f).read_text(), f


def test_discovery_records_parameterised_artifact(savings_cap):
    assert [s.action.type for s in savings_cap.steps] == ["click", "fill", "click", "extract"]
    assert "10042" not in savings_cap.to_json()
    assert savings_cap.inputs["member_id"].sensitivity.value == "pii"


def test_replay_success_other_member(env, savings_cap):
    res, _ = replay(env, savings_cap, {"member_id": "20077"})
    assert res.status == "succeeded" and res.outputs == {"savings_balance": "25.00"}


def test_business_outcomes_are_not_failures(env, savings_cap):
    res, _ = replay(env, savings_cap, {"member_id": "99999"})
    assert res.status == "business_outcome" and res.outcome.code == "member_not_found" and res.failure is None
    res, _ = replay(env, savings_cap, {"member_id": "30001"})
    assert res.outcome.code == "access_restricted"


def test_invalid_request_never_touches_ui(env, savings_cap):
    res, _ = replay(env, savings_cap, {"member_id": "12ab"})
    assert res.status == "failed" and res.failure.code == "invalid_request" and res.steps_completed == 0
    res, _ = replay(env, savings_cap.model_copy(update={"status": "draft"}), {"member_id": "10042"})
    assert res.failure.code == "invalid_request" and "draft" in res.failure.message


def test_recoverable_interstitial_and_session_expiry(env, savings_cap, faults):
    faults(interstitial=2)
    res, _ = replay(env, savings_cap, {"member_id": "10042"})
    assert res.status == "succeeded" and [r.code for r in res.recoveries] == ["system_notice", "system_notice"]
    faults(session_expire=1, after=2)
    res, _ = replay(env, savings_cap, {"member_id": "10042"})
    assert res.status == "succeeded" and [r.code for r in res.recoveries] == ["session_expired"]


def test_hard_failure_has_debuggable_evidence(env, savings_cap, faults):
    faults(abend=1, after=1)
    res, log = replay(env, savings_cap, {"member_id": "10042"})
    f = res.failure
    assert res.status == "failed" and f.code == "host_error" and "ABEND" in f.message
    assert f.step_id and f.observed.frames and (log.dir / f.evidence["screenshot"]).exists()
    dom = (log.dir / f.evidence["dom"]).read_text()
    assert "SYSTEM ERROR" in dom


def test_unknown_screen_without_escalation_fails_with_expected_vs_observed(env, savings_cap, faults):
    faults(survey=1, after=1)
    res, _ = replay(env, savings_cap, {"member_id": "10042"})
    assert res.failure.code == "checkpoint_timeout"
    assert "MEMBER INQUIRY" in res.failure.expected and "TELLER SURVEY" in json.dumps(res.failure.observed.frames)


def test_unknown_screen_human_takes_control_and_resumes(env, savings_cap, faults):
    faults(survey=1, after=1)
    op = SimulatedOperator([{"kind": "stuck", "actions": [{"type": "click", "frame": "main", "role": "button", "name": "Remind Me Later"}],
                             "decision": "resume"}])
    res, log = replay(env, savings_cap, {"member_id": "10042"}, operator=op, escalate=True)
    assert res.status == "succeeded" and len(res.interventions) == 1
    events = [json.loads(line) for line in (log.dir / "events.jsonl").read_text().splitlines()]
    kinds = [e["type"] for e in events]
    assert "human_action" in kinds and kinds.count("control_transferred") == 2
    human = next(e for e in events if e["type"] == "human_action")
    assert human["name"] == "Remind Me Later" and human["action"] == "click"


def test_cross_tenant_replay_with_caption_overlay(env, savings_cap):
    """Tenant B runs the same product with different captions; the same artifact runs via an alias overlay."""
    from target_app import server

    old = dict(server.LABELS)
    server.LABELS.update({"member_no": "Acct/Member #:", "search": "Find", "balance": "Current Bal"})
    try:
        res, _ = replay(env, savings_cap, {"member_id": "10042"})
        assert res.status == "failed" and res.failure.code == "target_not_found"  # without overlay: loud failure
        tb = env.tenant.model_copy(update={"aliases": {"Member Number": "Acct/Member #", "Search": "Find", "Balance": "Current Bal"}})
        res, _ = replay(env, savings_cap, {"member_id": "10042"}, tenant=tb)
        assert res.status == "succeeded" and res.outputs["savings_balance"] == "1234.56"
    finally:
        server.LABELS.update(old)


@pytest.fixture
def sub_cap(env, faults):
    log, surface, ctl = env.open("discover", operator=SimulatedOperator([{"kind": "approval", "claim": False, "decision": "approve"}]), escalate=True)
    session = Replayer(surface, env.profile, env.tenant, env.policy, log, ctl)
    params = {"member_id": "20077", "share_type": "HOLIDAY CLUB", "nickname": "XMAS FUND", "initial_deposit": "50.00"}
    out = Discovery(surface, ScriptedLLM(script("open_sub_account")), env.policy, log, ctl, session, []).run("open", params)
    assert out.status == "succeeded", out.reason
    return build_capability(out, capability_id="t.sub", title="t", description="d", goal_template="g", params=params,
                            profile=env.profile, run_id=log.run_id, model="scripted").model_copy(update={"status": "approved"})


def test_irreversible_step_requires_approval(env, sub_cap):
    assert sub_cap.risk == "irreversible" and sub_cap.inputs["share_type"].type == "enum"
    args = {"member_id": "10042", "share_type": "SAVINGS", "nickname": "RAINY DAY", "initial_deposit": "25"}
    deny = SimulatedOperator([{"kind": "approval", "claim": False, "decision": "deny", "note": "no"}])
    res, _ = replay(env, sub_cap, args, operator=deny, escalate=True)
    assert res.failure.code == "approval_denied"
    res, _ = replay(env, sub_cap, args, escalate=False)  # nobody to approve -> never executed silently
    assert res.failure.code == "escalation_aborted"
    ok = SimulatedOperator([{"kind": "approval", "claim": False, "decision": "approve"}])
    res, _ = replay(env, sub_cap, args, operator=ok, escalate=True)
    assert res.status == "succeeded" and len(res.outputs["confirmation_number"]) == 6


def test_host_validation_error_is_business_outcome(env, sub_cap):
    args = {"member_id": "10042", "share_type": "SAVINGS", "nickname": "BIG", "initial_deposit": "9999"}
    res, _ = replay(env, sub_cap, args)
    assert res.status == "business_outcome" and res.outcome.code == "invalid_input" and "ERR 1102" in res.outcome.message


def test_agent_cannot_leave_allowlist_or_touch_forbidden_controls(env, faults):
    log, surface, ctl = env.open("discover")
    session = Replayer(surface, env.profile, env.tenant, env.policy, log, ctl)
    llm = ScriptedLLM([
        {"tool": "click", "args": {"target": {"frame": "nav", "role": "link", "name": "Sign Off"}}},
        {"tool": "click", "args": {"target": {"frame": "nav", "role": "link", "name": "Member Inquiry"}}},
        {"tool": "request_help", "args": {"reason": "done testing"}},
    ])
    out = Discovery(surface, llm, env.policy, log, ctl, session, []).run("x", {})
    assert out.status == "failed"
    events = (log.dir / "events.jsonl").read_text()
    assert "forbidden control" in events and all(t.name != "Sign Off" for t in out.trace)
