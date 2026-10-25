"""REST API tests: auth/roles, registry lifecycle, invoke -> run, business outcomes, operator approval via API,
admission control and the audit trail. Uses the real (LLM-recorded) capability from evidence/."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from cua.api import create_app
from cua.db import Database
from cua.service import Settings
from tests.conftest import ROOT

SAVINGS = json.loads((ROOT / "evidence" / "capability.savings_balance.json").read_text())
SUB = json.loads((ROOT / "evidence" / "capability.open_sub_account.json").read_text())


@pytest.fixture
def settings(app_url, tmp_path, monkeypatch, faults):
    monkeypatch.setenv("T_USER", "TELLER01")
    monkeypatch.setenv("T_PASS", "teller01-local-test")
    tenants = tmp_path / "tenants"
    tenants.mkdir()
    (tenants / "test.json").write_text(json.dumps({
        "tenant_id": "test", "app_profile": "acme-coreteller@4.2", "base_url": app_url,
        "secrets": {"teller_user": "T_USER", "teller_password": "T_PASS"}}))
    policy = json.loads((ROOT / "policy.json").read_text())
    policy["allowed_origins"] = [app_url]
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    return Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}", runs_dir=tmp_path / "runs",
                    policy_path=tmp_path / "policy.json", tenants_dir=tenants, workers=2,
                    max_active_runs_per_tenant=2, handoff_timeout_s=60)


@pytest.fixture
def api(settings):
    db = Database(settings.database_url)
    db.create_all()
    keys = {role: db.create_api_key(f"{role}-1", role) for role in ("agent", "operator", "reviewer", "admin")}
    with TestClient(create_app(settings)) as client:
        client.keys = keys  # type: ignore[attr-defined]
        yield client


def h(api, role):
    return {"Authorization": f"Bearer {api.keys[role]}"}


def register_and_approve(api, body):
    r = api.post("/v1/capabilities", json=body, headers=h(api, "reviewer"))
    assert r.status_code == 201, r.text
    r = api.post(f"/v1/capabilities/{body['id']}/{body['version']}/status", json={"status": "approved"},
                 headers=h(api, "reviewer"))
    assert r.status_code == 200 and r.json()["reviewed_by"] == "reviewer-1"


def test_health_and_auth(api):
    assert api.get("/healthz").json() == {"status": "ok"}
    assert api.get("/readyz").json() == {"status": "ready"}
    assert api.get("/v1/capabilities").status_code == 401
    assert api.get("/v1/capabilities", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert api.post("/v1/capabilities", json=SAVINGS, headers=h(api, "agent")).status_code == 403
    assert api.get("/v1/interventions", headers=h(api, "agent")).status_code == 403


def test_dashboard_is_served(api):
    r = api.get("/")
    assert r.status_code == 200 and "<title>CUA Console</title>" in r.text
    assert api.get("/v1/me", headers=h(api, "operator")).json() == {"name": "operator-1", "role": "operator"}
    assert api.get("/v1/tenants", headers=h(api, "agent")).json() == ["test"]


def test_registry_lifecycle(api):
    r = api.post("/v1/capabilities", json=SAVINGS, headers=h(api, "reviewer"))
    assert r.status_code == 201 and r.json()["status"] == "draft" and len(r.json()["sha256"]) == 64
    assert api.post("/v1/capabilities", json=SAVINGS, headers=h(api, "reviewer")).status_code == 409  # immutable
    assert api.get("/v1/capabilities", headers=h(api, "agent")).json() == []  # agents never see drafts
    bad = SAVINGS | {"steps": SAVINGS["steps"][:1]}  # output without an extract step
    assert api.post("/v1/capabilities", json=bad | {"version": "9.9.9"}, headers=h(api, "reviewer")).status_code == 422
    api.post(f"/v1/capabilities/{SAVINGS['id']}/{SAVINGS['version']}/status", json={"status": "approved"},
             headers=h(api, "reviewer"))
    tools = api.get("/v1/capabilities", headers=h(api, "agent")).json()
    assert [t["name"] for t in tools] == ["coreteller__member__savings_balance"]
    assert tools[0]["parameters"]["required"] == ["member_id"]


def test_invoke_success_business_outcome_and_validation(api):
    register_and_approve(api, SAVINGS)
    cid = SAVINGS["id"]
    r = api.post(f"/v1/capabilities/{cid}/invoke", json={"tenant_id": "test", "inputs": {"member_id": "12ab"}},
                 headers=h(api, "agent"))
    assert r.status_code == 422 and "member_id" in r.json()["detail"]["invalid_inputs"]
    assert api.post(f"/v1/capabilities/{cid}/invoke", json={"tenant_id": "nope", "inputs": {"member_id": "1"}},
                    headers=h(api, "agent")).status_code == 404

    r = api.post(f"/v1/capabilities/{cid}/invoke?wait=90", json={"tenant_id": "test", "inputs": {"member_id": "20077"}},
                 headers=h(api, "agent"))
    run = r.json()
    assert run["status"] == "succeeded", run
    assert run["result"]["outputs"] == {"share_savings_balance": "25.00"}

    r = api.post(f"/v1/capabilities/{cid}/invoke?wait=90", json={"tenant_id": "test", "inputs": {"member_id": "99999"}},
                 headers=h(api, "agent"))
    assert r.json()["status"] == "business_outcome" and r.json()["result"]["outcome"]["code"] == "member_not_found"

    runs = api.get("/v1/runs", headers=h(api, "operator")).json()
    assert {x["status"] for x in runs} == {"succeeded", "business_outcome"}
    assert "99999" not in json.dumps(runs)  # pii input never persisted in clear


def test_irreversible_step_approved_by_operator_via_api(api):
    register_and_approve(api, SUB)
    inputs = {"member_id": "10042", "share_type": "SAVINGS", "nickname": "RAINY DAY", "initial_deposit": "25"}
    r = api.post(f"/v1/capabilities/{SUB['id']}/invoke", json={"tenant_id": "test", "inputs": inputs}, headers=h(api, "agent"))
    run_id = r.json()["run_id"]
    req = None
    for _ in range(240):
        reqs = api.get("/v1/interventions", headers=h(api, "operator")).json()
        if reqs:
            req = reqs[0]
            break
        time.sleep(0.25)
    assert req and req["kind"] == "approval" and req["run_id"] == run_id and req["control"]["holder"] == "automation"
    shot = api.get(f"/v1/runs/{run_id}/evidence/{req['screenshot']}", headers=h(api, "operator"))
    assert shot.status_code == 200 and shot.headers["content-type"] == "image/png"
    assert api.get(f"/v1/runs/{run_id}/evidence/../../policy.json", headers=h(api, "operator")).status_code == 404

    r = api.post(f"/v1/interventions/{req['id']}/resolve", json={"decision": "approve", "note": "checked"},
                 headers=h(api, "operator"))
    assert r.status_code == 200 and r.json()["decision"] == "approve"
    assert api.post(f"/v1/interventions/{req['id']}/resolve", json={"decision": "deny"},
                    headers=h(api, "operator")).status_code == 409  # exactly once
    for _ in range(240):
        run = api.get(f"/v1/runs/{run_id}", headers=h(api, "agent")).json()
        if run["status"] not in ("queued", "running"):
            break
        time.sleep(0.25)
    assert run["status"] == "succeeded" and len(run["result"]["outputs"]["confirmation_number"]) == 6

    actions = [a["action"] for a in api.get("/v1/audit", headers=h(api, "admin")).json()]
    assert {"capability.register", "capability.approved", "capability.invoke", "intervention.approve"} <= set(actions)


def test_admission_control_per_tenant(settings):
    settings.max_active_runs_per_tenant = 0
    db = Database(settings.database_url)
    db.create_all()
    agent, reviewer = db.create_api_key("a", "agent"), db.create_api_key("r", "reviewer")
    with TestClient(create_app(settings)) as c:
        c.post("/v1/capabilities", json=SAVINGS, headers={"Authorization": f"Bearer {reviewer}"})
        c.post(f"/v1/capabilities/{SAVINGS['id']}/{SAVINGS['version']}/status", json={"status": "approved"},
               headers={"Authorization": f"Bearer {reviewer}"})
        r = c.post(f"/v1/capabilities/{SAVINGS['id']}/invoke", json={"tenant_id": "test", "inputs": {"member_id": "1"}},
                   headers={"Authorization": f"Bearer {agent}"})
        assert r.status_code == 429
