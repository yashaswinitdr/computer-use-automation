"""Service layer: configuration, the replay worker pool, and run bookkeeping. The HTTP API is a thin shell over this.

Replays run in a bounded thread pool. Each job owns its own Playwright instance and browser for its whole life
(sync Playwright is per-thread), so jobs are isolated: a crashed browser fails one run, not the service.
"""

from __future__ import annotations

import os
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cua import catalog as cat
from cua.db import Database, utcnow
from cua.evidence import RunLog, new_run_id
from cua.handoff import Controller
from cua.handoff_db import DbHandoffStore
from cua.policy import Policy
from cua.redact import Redactor
from cua.schema import Capability, TenantConfig

ROOT = cat.ROOT


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.environ.get("CUA_DATABASE_URL", f"sqlite:///{ROOT / 'cua.db'}"))
    runs_dir: Path = field(default_factory=lambda: Path(os.environ.get("CUA_RUNS_DIR", ROOT / "runs")))
    policy_path: Path = field(default_factory=lambda: Path(os.environ.get("CUA_POLICY", ROOT / "policy.json")))
    tenants_dir: Path = field(default_factory=lambda: Path(os.environ.get("CUA_TENANTS_DIR", ROOT / "tenants")))
    workers: int = field(default_factory=lambda: int(os.environ.get("CUA_WORKERS", "2")))
    max_active_runs_per_tenant: int = field(default_factory=lambda: int(os.environ.get("CUA_MAX_RUNS_PER_TENANT", "2")))
    handoff_timeout_s: float = field(default_factory=lambda: float(os.environ.get("CUA_HANDOFF_TIMEOUT", "300")))
    bootstrap_admin_key: str | None = field(default_factory=lambda: os.environ.get("CUA_BOOTSTRAP_ADMIN_KEY"))


class TooManyRuns(Exception):
    pass


class Service:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.db = Database(self.settings.database_url)
        self.db.create_all()
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)
        self.store = DbHandoffStore(self.db, self.settings.runs_dir)
        self.policy = Policy.load(self.settings.policy_path)
        self._pool = ThreadPoolExecutor(max_workers=self.settings.workers, thread_name_prefix="replay")
        self._admit = threading.Lock()
        # Unredacted outputs are handed to the caller once and never persisted (bounded, in memory).
        self._last_outputs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._bootstrap()

    def _bootstrap(self) -> None:
        key = self.settings.bootstrap_admin_key
        if key and self.db.lookup_api_key(key) is None:
            from cua.db import ApiKeyRow

            with self.db.session() as s, s.begin():
                s.add(ApiKeyRow(key_hash=self.db._hash(key), prefix=key[:12], name="bootstrap-admin", role="admin"))

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------------ lookups
    def tenant(self, tenant_id: str) -> TenantConfig:
        p = self.settings.tenants_dir / f"{tenant_id}.json"
        if not p.is_file() or p.parent.resolve() != self.settings.tenants_dir.resolve():
            raise KeyError(f"unknown tenant {tenant_id!r}")
        return cat.load_tenant(str(p))

    def capability(self, cid: str, version: str | None, allow_draft: bool) -> Capability:
        row = self.db.get_capability(cid, version, approved_only=not allow_draft)
        if row is None:
            raise KeyError(f"no {'approved ' if not allow_draft else ''}capability {cid}{'@' + version if version else ''}")
        return Capability.model_validate(row.body)

    # ------------------------------------------------------------------ invoke
    def submit_replay(self, cap: Capability, tenant: TenantConfig, inputs: dict[str, Any], actor: str,
                      allow_draft: bool = False) -> str:
        with self._admit:  # admission control: bounded concurrent sessions per tenant (host systems are fragile)
            if self.db.count_active_runs(tenant.tenant_id) >= self.settings.max_active_runs_per_tenant:
                raise TooManyRuns(f"tenant {tenant.tenant_id} already has "
                                  f"{self.settings.max_active_runs_per_tenant} active runs")
            run_id = new_run_id("replay")
            self.db.create_run(run_id, "replay", cap.ref, tenant.tenant_id, actor)
        self._pool.submit(self._run_replay, run_id, cap, tenant, inputs, allow_draft)
        return run_id

    def _run_replay(self, run_id: str, cap: Capability, tenant: TenantConfig, inputs: dict[str, Any],
                    allow_draft: bool) -> None:
        from cua.replay import Replayer
        from cua.surface.web import WebSurface

        self.db.update_run(run_id, status="running", started_at=utcnow())
        log = RunLog(self.settings.runs_dir, run_id, Redactor(), echo=False)
        surface = None
        try:
            profile = cat.load_profile(tenant.app_profile)
            surface = WebSurface(tenant.base_url, self.policy)
            ctl = Controller(self.store, run_id, log, surface, timeout_s=self.settings.handoff_timeout_s,
                             pii_labels=profile.pii_labels)
            res = Replayer(surface, profile, tenant, self.policy, log, ctl).run(cap, inputs, allow_draft=allow_draft)
            redacted = res.model_dump(mode="json")
            redacted["outputs"] = Replayer._loggable_outputs(cap, res.outputs)
            self.db.update_run(run_id, status=res.status, result=redacted, finished_at=utcnow())
            self._last_outputs[run_id] = res.outputs
            while len(self._last_outputs) > 1000:
                self._last_outputs.popitem(last=False)
        except Exception as e:  # infrastructure failure (browser crash, bad config): never leave a run "running"
            log.event("run_crashed", error=repr(e), trace=traceback.format_exc()[-2000:])
            self.db.update_run(run_id, status="failed", error=f"{type(e).__name__}: {e}"[:2000], finished_at=utcnow())
        finally:
            if surface is not None:
                surface.close()
            log.close()

    def take_outputs(self, run_id: str) -> dict[str, Any] | None:
        return self._last_outputs.pop(run_id, None)
