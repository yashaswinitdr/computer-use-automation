"""Persistence for the service: capability registry, runs, interventions, control transfers, API keys, audit log.

SQLAlchemy 2.x; SQLite for local development, PostgreSQL in docker-compose (``CUA_DATABASE_URL``). Large
evidence files (screenshots, DOM digests, event logs) stay on disk / object storage; rows point at them.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    select,
    update,
)
from sqlalchemy.engine import CursorResult, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class CapabilityRow(Base):
    __tablename__ = "capabilities"
    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    version: Mapped[str] = mapped_column(String(40), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    sha256: Mapped[str] = mapped_column(String(64))
    body: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[str | None] = mapped_column(String(100))
    reviewed_by: Mapped[str | None] = mapped_column(String(100))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RunRow(Base):
    __tablename__ = "runs"
    run_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))
    capability: Mapped[str] = mapped_column(String(250), index=True)
    tenant_id: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)  # queued|running|succeeded|business_outcome|failed
    requested_by: Mapped[str | None] = mapped_column(String(100))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # redacted ReplayResult
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class InterventionRow(Base):
    __tablename__ = "interventions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    body: Mapped[dict[str, Any]] = mapped_column(JSON)  # full InterventionRequest
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ControlRow(Base):
    __tablename__ = "control_state"
    run_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    holder: Mapped[str] = mapped_column(String(20))
    epoch: Mapped[int] = mapped_column(Integer)
    operator: Mapped[str | None] = mapped_column(String(100))
    since: Mapped[str] = mapped_column(String(40))
    reason: Mapped[str] = mapped_column(String(300))


class ControlTransferRow(Base):
    __tablename__ = "control_transfers"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(80), ForeignKey("control_state.run_id"), index=True)
    holder: Mapped[str] = mapped_column(String(20))
    epoch: Mapped[int] = mapped_column(Integer)
    operator: Mapped[str | None] = mapped_column(String(100))
    since: Mapped[str] = mapped_column(String(40))
    reason: Mapped[str] = mapped_column(String(300))


class ApiKeyRow(Base):
    __tablename__ = "api_keys"
    key_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    prefix: Mapped[str] = mapped_column(String(12))
    name: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditRow(Base):
    __tablename__ = "audit_log"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20))
    action: Mapped[str] = mapped_column(String(60))
    target: Mapped[str] = mapped_column(String(250))
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON)


# --------------------------------------------------------------------------------------------- engine/session
class Database:
    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get("CUA_DATABASE_URL", "sqlite:///./cua.db")
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if self.url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine: Engine = create_engine(self.url, **kwargs)
        if self.url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn, _rec):  # WAL: concurrent readers while a worker writes
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=5000")
                cur.close()
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._sessions()

    def ping(self) -> bool:
        with self.engine.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return True

    # ------------------------------------------------------------------ api keys
    @staticmethod
    def _hash(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def create_api_key(self, name: str, role: str) -> str:
        key = "cua_" + secrets.token_urlsafe(32)
        with self.session() as s, s.begin():
            s.add(ApiKeyRow(key_hash=self._hash(key), prefix=key[:12], name=name, role=role))
        return key  # shown once; only the hash is stored

    def lookup_api_key(self, key: str) -> ApiKeyRow | None:
        with self.session() as s:
            row = s.get(ApiKeyRow, self._hash(key))
            return row if row and not row.revoked else None

    def revoke_api_key(self, prefix: str) -> int:
        with self.session() as s, s.begin():
            res = cast(CursorResult[Any], s.execute(update(ApiKeyRow).where(ApiKeyRow.prefix == prefix).values(revoked=True)))
            return res.rowcount

    # ------------------------------------------------------------------ audit
    def audit(self, actor: str, role: str, action: str, target: str, detail: dict[str, Any] | None = None) -> None:
        with self.session() as s, s.begin():
            s.add(AuditRow(actor=actor, role=role, action=action, target=target, detail=detail))

    def audit_log(self, limit: int = 100) -> list[AuditRow]:
        with self.session() as s:
            return list(s.scalars(select(AuditRow).order_by(AuditRow.seq.desc()).limit(limit)))

    # ------------------------------------------------------------------ capability registry
    def put_capability(self, body: dict[str, Any], created_by: str) -> CapabilityRow:
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        row = CapabilityRow(id=body["id"], version=body["version"], status=body.get("status", "draft"),
                            sha256=hashlib.sha256(canonical.encode()).hexdigest(), body=body, created_by=created_by)
        with self.session() as s, s.begin():
            if s.get(CapabilityRow, (row.id, row.version)) is not None:
                raise ValueError(f"{row.id}@{row.version} already registered; versions are immutable")
            s.add(row)
        return row

    def get_capability(self, cid: str, version: str | None = None, approved_only: bool = False) -> CapabilityRow | None:
        with self.session() as s:
            q = select(CapabilityRow).where(CapabilityRow.id == cid)
            if version:
                q = q.where(CapabilityRow.version == version)
            if approved_only:
                q = q.where(CapabilityRow.status == "approved")
            rows = list(s.scalars(q))
        if not rows:
            return None
        return max(rows, key=lambda r: tuple(int(x) for x in r.version.split(".")))

    def list_capabilities(self) -> list[CapabilityRow]:
        with self.session() as s:
            return list(s.scalars(select(CapabilityRow).order_by(CapabilityRow.id, CapabilityRow.version)))

    def set_capability_status(self, cid: str, version: str, status: str, reviewer: str) -> CapabilityRow:
        with self.session() as s, s.begin():
            row = s.get(CapabilityRow, (cid, version))
            if row is None:
                raise KeyError(f"{cid}@{version}")
            row.status = status
            row.body = {**row.body, "status": status}
            if status == "approved":
                row.reviewed_by, row.reviewed_at = reviewer, utcnow()
                prov = dict(row.body.get("provenance", {}))
                prov.update(reviewed_by=reviewer, reviewed_at=row.reviewed_at.isoformat())
                row.body = {**row.body, "provenance": prov}
            return row

    # ------------------------------------------------------------------ runs
    def create_run(self, run_id: str, kind: str, capability: str, tenant_id: str, requested_by: str) -> None:
        with self.session() as s, s.begin():
            s.add(RunRow(run_id=run_id, kind=kind, capability=capability, tenant_id=tenant_id, status="queued",
                         requested_by=requested_by))

    def update_run(self, run_id: str, **fields: Any) -> None:
        with self.session() as s, s.begin():
            s.execute(update(RunRow).where(RunRow.run_id == run_id).values(**fields))

    def get_run(self, run_id: str) -> RunRow | None:
        with self.session() as s:
            return s.get(RunRow, run_id)

    def list_runs(self, status: str | None = None, limit: int = 50) -> list[RunRow]:
        with self.session() as s:
            q = select(RunRow).order_by(RunRow.created_at.desc()).limit(limit)
            if status:
                q = q.where(RunRow.status == status)
            return list(s.scalars(q))

    def count_active_runs(self, tenant_id: str) -> int:
        with self.session() as s:
            return len(list(s.scalars(select(RunRow.run_id).where(
                RunRow.tenant_id == tenant_id, RunRow.status.in_(("queued", "running"))))))


def iter_rows(rows: list[Any]) -> Iterator[dict[str, Any]]:
    for r in rows:
        yield {c.name: getattr(r, c.name) for c in r.__table__.columns}
