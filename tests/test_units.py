"""Fast unit tests: schema contract, redaction, policy, recorder parameterisation, handoff state machine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cua.agent.discovery import DiscoveryOutcome, TraceEntry
from cua.catalog import load_profile, tool_definition
from cua.handoff import HandoffStore, InterventionRequest
from cua.policy import Policy
from cua.recorder import build_capability, canonicalize
from cua.redact import Redactor
from cua.replay import coerce_output
from cua.schema import Capability, OutputSpec, ParamSpec
from tests.conftest import ROOT

MIN_CAP = {
    "id": "x.y", "version": "1.0.0", "title": "t", "description": "d",
    "app": {"profile": "p", "profile_version": "1"},
    "steps": [{"id": "s1", "intent": "i", "action": {"type": "fill", "value": {"kind": "param", "name": "member_id"},
               "target": {"frame": "main", "role": "textbox", "name": "Member Number"}}}],
    "success": {"all_of": [{"kind": "title", "frame": "main", "equals": "X"}]},
    "provenance": {"method": "authored"},
}


def test_capability_rejects_undeclared_param():
    with pytest.raises(ValidationError, match="undeclared input 'member_id'"):
        Capability.model_validate(MIN_CAP)


def test_capability_requires_extract_step_per_output():
    bad = MIN_CAP | {"inputs": {"member_id": {"type": "string"}}, "outputs": {"bal": {"type": "money"}}}
    with pytest.raises(ValidationError, match="exactly one extract step"):
        Capability.model_validate(bad)


def test_capability_round_trip_and_risk():
    cap = Capability.model_validate(MIN_CAP | {"inputs": {"member_id": {"type": "string"}}})
    again = Capability.model_validate_json(cap.to_json())
    assert again == cap and again.risk == "read" and '"schema": "cua.capability/v1"' in cap.to_json()


def test_param_validation():
    spec = ParamSpec(type="string", pattern=r"^\d{1,12}$")
    assert spec.validate_value(" 10042 ") == "10042"
    with pytest.raises(ValueError):
        spec.validate_value("12ab")
    with pytest.raises(ValueError):
        ParamSpec(type="decimal", maximum=5000).validate_value("9999")
    with pytest.raises(ValueError):
        ParamSpec(type="enum", enum=["SAVINGS"]).validate_value("CHECKING")


def test_coerce_money():
    assert coerce_output(OutputSpec(type="money"), "$1,234.5") == "1234.50"
    assert coerce_output(OutputSpec(type="money"), "(12.00)") == "-12.00"
    with pytest.raises(ValueError):
        coerce_output(OutputSpec(type="money"), "N/A")


def test_redaction_patterns_and_values():
    r = Redactor()
    r.add_secret("hunter2-secret")
    r.add_pii("10042")
    s = r.text("pw=hunter2-secret ssn 123-45-6789 card 4111 1111 1111 1111 mail a.b@bank.com member 10042")
    assert "hunter2" not in s and "123-45-6789" not in s and "4111" not in s and "a.b@bank.com" not in s
    assert "10042" not in s and "***42" in s
    # things that merely look numeric must survive
    assert r.text("run discover-20260926-150616 cap x.y@1.0.0.json") == "run discover-20260926-150616 cap x.y@1.0.0.json"
    assert r.obj({"password": "x", "n": ["123-45-6789"]}) == {"password": "[SECRET]", "n": ["[SSN]"]}


def test_policy_urls_and_risk():
    p = Policy.load(ROOT / "policy.json")
    assert p.url_allowed("http://127.0.0.1:5055/mbr/10042")[0]
    assert not p.url_allowed("http://127.0.0.1:5055/__faults?x=1")[0]
    assert not p.url_allowed("http://127.0.0.1:5055/logout")[0]
    assert not p.url_allowed("https://evil.example.com/")[0]
    assert p.classify("click", "Confirm") == "irreversible"
    assert p.classify("click", "Continue") == "write"
    assert p.classify("click", "Search") == "read"
    assert p.classify("click", "Sign Off") is None
    assert p.classify("fill", "Nickname") == "read"


def test_recorder_parameterises_values_routes_and_rows():
    params = {"member_id": "10042"}
    before = [("nav", "/nav", "MENU"), ("main", "/inq", "MEMBER INQUIRY")]
    after = [("nav", "/nav", "MENU"), ("main", "/mbr/10042", "MEMBER SUMMARY")]
    trace = [
        TraceEntry("model", "fill", "main", "textbox", "Member Number", css='input[name="q"]', text="10042", before=before, after=before),
        TraceEntry("model", "click", "main", "button", "Search", css="body > table:nth-of-type(2) > tr", before=before, after=after),
        TraceEntry("model", "extract", extract={"output": "bal", "row_key_param": None, "source": {
            "kind": "table_cell", "frame": "main", "row_key_column": "Description", "row_key": {"kind": "literal", "value": "SHARE SAVINGS"},
            "column": "Balance"}}),
    ]
    out = DiscoveryOutcome("succeeded", "", trace, {"bal": "1.00"}, {"bal": OutputSpec(type="money")}, "SHARE SAVINGS", after, 3)
    cap = build_capability(out, capability_id="t.cap", title="t", description="d", goal_template="g {member_id}",
                           params=params, profile=load_profile("acme-coreteller@4.2"), run_id="r", model="m")
    blob = cap.to_json()
    assert "10042" not in blob  # example values never leak into the artifact
    assert cap.steps[0].action.value.kind == "param" and cap.steps[0].expect is None
    assert cap.steps[0].action.target.fallbacks[0].value == 'input[name="q"]'
    assert cap.steps[1].action.target.fallbacks == []  # positional CSS is deliberately not kept
    assert cap.steps[1].expect.all_of[0].pattern == "/mbr/{member_id}"
    assert {o.code for o in cap.outcomes} == {"member_not_found", "access_restricted", "invalid_input"}
    assert tool_definition(cap)["parameters"]["required"] == ["member_id"]
    assert canonicalize("/mbr/10042/x", params) == "/mbr/{member_id}/x"


def test_handoff_control_transfers(tmp_path):
    store = HandoffStore(tmp_path)
    assert store.control("run1").holder == "automation"
    store.open_request(InterventionRequest(id="req-1", run_id="run1", kind="approval", reason="r", subject="s",
                                           allowed_decisions=["approve", "deny"]))
    store.claim("run1", "req-1", "alice")
    c = store.control("run1")
    assert (c.holder, c.operator, c.epoch) == ("human", "alice", 1)
    with pytest.raises(ValueError):
        store.claim("run1", "req-1", "bob")  # already claimed
    with pytest.raises(ValueError):
        store.resolve("run1", "req-1", "resume", "alice")  # not an allowed decision for this request
    store.resolve("run1", "req-1", "approve", "alice", "ok")
    c = store.control("run1")
    assert (c.holder, c.epoch) == ("automation", 2)
    assert store.list() == []
