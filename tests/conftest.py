from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest
from werkzeug.serving import make_server

from cua import catalog as cat
from cua.evidence import RunLog, new_run_id
from cua.handoff import Controller, HandoffStore
from cua.policy import Policy
from cua.redact import Redactor
from cua.schema import TenantConfig

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def app_url():
    from target_app import server

    srv = make_server("127.0.0.1", 0, server.app, threaded=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def faults(app_url):
    def set_(**kw):
        q = "&".join(f"{k}={v}" for k, v in {"reset": 1, **kw}.items())
        return json.loads(urllib.request.urlopen(f"{app_url}/__faults?{q}").read())

    set_()
    yield set_
    set_()


@pytest.fixture
def env(app_url, tmp_path, monkeypatch):
    """Everything needed to run discovery/replay against the in-process target app."""
    monkeypatch.setenv("T_USER", "TELLER01")
    monkeypatch.setenv("T_PASS", "teller01-local-test")
    base = Policy.load(ROOT / "policy.json")
    policy = base.model_copy(update={"allowed_origins": [app_url]})
    tenant = TenantConfig(tenant_id="test", app_profile="acme-coreteller@4.2", base_url=app_url,
                          secrets={"teller_user": "T_USER", "teller_password": "T_PASS"})
    profile = cat.load_profile(tenant.app_profile)
    created = []

    class Env:
        def __init__(self):
            self.policy, self.tenant, self.profile, self.runs = policy, tenant, profile, tmp_path / "runs"

        def open(self, kind="test", operator=None, escalate=False, tenant=None):
            from cua.surface.web import WebSurface

            while created:  # one sync Playwright per thread: close the previous surface first
                sfc, lg = created.pop()
                sfc.close()
                lg.close()
            t = tenant or self.tenant
            log = RunLog(self.runs, new_run_id(kind), Redactor(), echo=False)
            surface = WebSurface(t.base_url, self.policy)
            ctl = Controller(HandoffStore(self.runs), log.run_id, log, surface, operator=operator, timeout_s=30,
                             pii_labels=self.profile.pii_labels)
            ctl.enabled = escalate
            created.append((surface, log))
            return log, surface, ctl

    yield Env()
    for surface, log in created:
        surface.close()
        log.close()
