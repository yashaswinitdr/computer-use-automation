"""REST API: the production entry point for agents (invoke capabilities), reviewers (registry) and operators (handoff).

Auth: ``Authorization: Bearer <api key>``. Keys are stored hashed and carry one role:
  agent    -- list/invoke approved capabilities, read its runs
  operator -- read runs, list/claim/resolve intervention requests, fetch evidence
  reviewer -- register capability versions, approve/deprecate them
  admin    -- everything
Every state-changing call is written to the audit log.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ValidationError
from sqlalchemy import select

from cua.catalog import tool_definition
from cua.db import InterventionRow
from cua.schema import Capability
from cua.service import Service, Settings, TooManyRuns

WEB = Path(__file__).parent / "web"
ROLES = ("agent", "operator", "reviewer", "admin")


class Principal(BaseModel):
    name: str
    role: str


class InvokeRequest(BaseModel):
    tenant_id: str
    inputs: dict[str, Any] = {}
    version: str | None = None
    allow_draft: bool = False


class StatusChange(BaseModel):
    status: Literal["approved", "deprecated", "draft"]


class Resolution(BaseModel):
    decision: Literal["approve", "deny", "resume", "abort"]
    note: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    holder: dict[str, Service] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        holder["svc"] = Service(settings)
        yield
        holder["svc"].shutdown()

    app = FastAPI(title="Computer-Use Automation API", version="1.1.0", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    def svc() -> Service:
        return holder["svc"]

    def principal(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
        if creds is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
        row = svc().db.lookup_api_key(creds.credentials)
        if row is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or revoked API key")
        return Principal(name=row.name, role=row.role)

    def require(*roles: str):
        def dep(p: Principal = Depends(principal)) -> Principal:
            if p.role != "admin" and p.role not in roles:
                raise HTTPException(status.HTTP_403_FORBIDDEN, f"role '{p.role}' may not do this")
            return p
        return dep

    # ------------------------------------------------------------------ health
    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    def readyz() -> dict[str, str]:
        try:
            svc().db.ping()
        except Exception as e:  # report, don't crash the probe
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"database unavailable: {type(e).__name__}") from e
        return {"status": "ready"}

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(WEB / "index.html")

    @app.get("/v1/me", tags=["ops"])
    def me(p: Principal = Depends(principal)) -> Principal:
        return p

    @app.get("/v1/tenants", tags=["ops"])
    def tenants(p: Principal = Depends(principal)) -> list[str]:
        return sorted(f.stem for f in svc().settings.tenants_dir.glob("*.json"))

    # ------------------------------------------------------------------ capability registry
    @app.get("/v1/capabilities", tags=["capabilities"])
    def list_capabilities(include_drafts: bool = False, p: Principal = Depends(principal)) -> list[dict[str, Any]]:
        """Approved capabilities as agent tool definitions (latest version of each)."""
        latest: dict[str, Capability] = {}
        for row in svc().db.list_capabilities():
            if row.status == "deprecated" or (row.status == "draft" and not (include_drafts and p.role != "agent")):
                continue
            cap = Capability.model_validate(row.body)
            cur = latest.get(cap.id)
            if cur is None or tuple(map(int, cap.version.split("."))) > tuple(map(int, cur.version.split("."))):
                latest[cap.id] = cap
        return [tool_definition(c) | {"status": c.status} for c in latest.values()]

    @app.get("/v1/capabilities/{cid}", tags=["capabilities"])
    def get_capability(cid: str, version: str | None = None, p: Principal = Depends(principal)) -> dict[str, Any]:
        row = svc().db.get_capability(cid, version, approved_only=p.role == "agent")
        if row is None:
            raise HTTPException(404, "capability not found")
        return {"sha256": row.sha256, "status": row.status, "capability": row.body}

    @app.post("/v1/capabilities", status_code=201, tags=["capabilities"])
    def register_capability(body: dict[str, Any], p: Principal = Depends(require("reviewer"))) -> dict[str, Any]:
        """Register a new, immutable capability version (always starts as draft)."""
        try:
            cap = Capability.model_validate(body | {"status": "draft"})
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False, include_context=False)) from e
        problems = svc().policy.check_capability(cap)
        if problems:
            raise HTTPException(422, {"policy_violations": problems})
        try:
            row = svc().db.put_capability(cap.model_dump(mode="json", by_alias=True, exclude_none=True), p.name)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        svc().db.audit(p.name, p.role, "capability.register", cap.ref, {"sha256": row.sha256})
        return {"capability": cap.ref, "status": "draft", "sha256": row.sha256}

    @app.post("/v1/capabilities/{cid}/{version}/status", tags=["capabilities"])
    def set_status(cid: str, version: str, change: StatusChange, p: Principal = Depends(require("reviewer"))) -> dict[str, Any]:
        try:
            row = svc().db.set_capability_status(cid, version, change.status, p.name)
        except KeyError as e:
            raise HTTPException(404, "capability not found") from e
        svc().db.audit(p.name, p.role, f"capability.{change.status}", f"{cid}@{version}", {"sha256": row.sha256})
        return {"capability": f"{cid}@{version}", "status": row.status, "reviewed_by": row.reviewed_by}

    # ------------------------------------------------------------------ invoke + runs
    @app.post("/v1/capabilities/{cid}/invoke", status_code=202, tags=["runs"])
    def invoke(cid: str, req: InvokeRequest, wait: float = Query(0, ge=0, le=120,
               description="Seconds to wait for completion before returning (0 = return immediately)"),
               p: Principal = Depends(require("agent"))) -> dict[str, Any]:
        s = svc()
        if req.allow_draft and p.role != "admin":
            raise HTTPException(403, "only admins may invoke draft capabilities")
        try:
            cap = s.capability(cid, req.version, req.allow_draft)
            tenant = s.tenant(req.tenant_id)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\"")) from e
        errors = {}
        for name, spec in cap.inputs.items():  # fail fast with 422 instead of queueing a doomed run
            if name in req.inputs:
                try:
                    spec.validate_value(req.inputs[name])
                except ValueError as e:
                    errors[name] = str(e)
            elif spec.required:
                errors[name] = "required"
        errors |= {k: "unknown input" for k in set(req.inputs) - set(cap.inputs)}
        if errors:
            raise HTTPException(422, {"invalid_inputs": errors})
        try:
            run_id = s.submit_replay(cap, tenant, req.inputs, p.name, allow_draft=req.allow_draft)
        except TooManyRuns as e:
            raise HTTPException(429, str(e)) from e
        s.db.audit(p.name, p.role, "capability.invoke", cap.ref, {"run_id": run_id, "tenant": tenant.tenant_id,
                                                                     "inputs": sorted(req.inputs)})
        deadline = time.monotonic() + wait
        while wait and time.monotonic() < deadline:
            run = s.db.get_run(run_id)
            if run and run.status not in ("queued", "running"):
                return _run_view(s, run, reveal=True)
            time.sleep(0.25)
        return {"run_id": run_id, "status": "queued", "links": {"self": f"/v1/runs/{run_id}"}}

    @app.get("/v1/runs/{run_id}", tags=["runs"])
    def get_run(run_id: str, p: Principal = Depends(principal)) -> dict[str, Any]:
        run = svc().db.get_run(run_id)
        if run is None or (p.role == "agent" and run.requested_by != p.name):
            raise HTTPException(404, "run not found")
        return _run_view(svc(), run, reveal=p.role == "agent")

    @app.get("/v1/runs", tags=["runs"])
    def list_runs(status_: str | None = Query(None, alias="status"), limit: int = Query(50, le=500),
                  p: Principal = Depends(require("operator", "reviewer"))) -> list[dict[str, Any]]:
        return [_run_view(svc(), r, reveal=False) for r in svc().db.list_runs(status_, limit)]

    @app.get("/v1/runs/{run_id}/files", tags=["runs"])
    def run_files(run_id: str, p: Principal = Depends(require("operator", "reviewer"))) -> list[str]:
        base = svc().settings.runs_dir / run_id
        return sorted(str(f.relative_to(base)).replace("\\", "/") for f in base.glob("screenshots/*.png"))

    @app.get("/v1/runs/{run_id}/evidence/{path:path}", tags=["runs"])
    def evidence(run_id: str, path: str, p: Principal = Depends(require("operator", "reviewer"))) -> FileResponse:
        base = (svc().settings.runs_dir / run_id).resolve()
        target = (base / path).resolve()
        if not target.is_relative_to(base) or not target.is_file():  # no path traversal
            raise HTTPException(404, "not found")
        return FileResponse(target)

    # ------------------------------------------------------------------ human handoff
    @app.get("/v1/interventions", tags=["handoff"])
    def interventions(all_: bool = Query(False, alias="all"), p: Principal = Depends(require("operator"))) -> list[dict[str, Any]]:
        reqs = svc().store.list(status=None if all_ else "open")
        return [r.model_dump(mode="json") | {"control": svc().store.control(r.run_id).model_dump()} for r in reqs]

    def _find(req_id: str) -> InterventionRow:
        with svc().db.session() as s:
            row = s.scalars(select(InterventionRow).where(InterventionRow.id == req_id)).first()
        if row is None:
            raise HTTPException(404, "intervention request not found")
        return row

    @app.post("/v1/interventions/{req_id}/claim", tags=["handoff"])
    def claim(req_id: str, p: Principal = Depends(require("operator"))) -> dict[str, Any]:
        row = _find(req_id)
        try:
            req = svc().store.claim(row.run_id, req_id, p.name)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        svc().db.audit(p.name, p.role, "intervention.claim", req_id, {"run_id": row.run_id})
        return {"request": req.id, "status": req.status, "control": svc().store.control(row.run_id).model_dump()}

    @app.post("/v1/interventions/{req_id}/resolve", tags=["handoff"])
    def resolve(req_id: str, r: Resolution, p: Principal = Depends(require("operator"))) -> dict[str, Any]:
        row = _find(req_id)
        try:
            req = svc().store.resolve(row.run_id, req_id, r.decision, p.name, r.note)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        svc().db.audit(p.name, p.role, f"intervention.{r.decision}", req_id, {"run_id": row.run_id, "note": r.note})
        return {"request": req.id, "status": req.status, "decision": req.decision,
                "control": svc().store.control(row.run_id).model_dump()}

    # ------------------------------------------------------------------ audit
    @app.get("/v1/audit", tags=["ops"])
    def audit(limit: int = Query(100, le=1000), p: Principal = Depends(require("admin"))) -> list[dict[str, Any]]:
        return [{"seq": a.seq, "at": a.at, "actor": a.actor, "role": a.role, "action": a.action,
                 "target": a.target, "detail": a.detail} for a in svc().db.audit_log(limit)]

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # never leak stack traces to clients
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"detail": "internal error"})

    return app


def _run_view(s: Service, run, reveal: bool) -> dict[str, Any]:
    view = {"run_id": run.run_id, "capability": run.capability, "tenant_id": run.tenant_id, "status": run.status,
            "requested_by": run.requested_by, "created_at": run.created_at, "started_at": run.started_at,
            "finished_at": run.finished_at, "result": run.result, "error": run.error}
    if reveal and run.result and run.status == "succeeded":
        outputs = s.take_outputs(run.run_id)
        if outputs is not None:  # the caller gets clear outputs exactly once; persisted copy stays redacted
            view["result"] = {**run.result, "outputs": outputs}
    return view
