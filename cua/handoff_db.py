"""Database-backed handoff store: same interface as ``HandoffStore`` (file based), safe for concurrent operators.

Claiming and resolving are single conditional UPDATEs (``... WHERE status = 'open'``), so two operators racing to
claim the same request cannot both win, and a request can only be resolved once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult

from cua.db import ControlRow, ControlTransferRow, Database, InterventionRow
from cua.handoff import ControlState, Decision, InterventionRequest, _now


class DbHandoffStore:
    def __init__(self, db: Database, runs_root: Path):
        self.db = db
        self.root = Path(runs_root)  # evidence files (screenshots, DOM digests) stay on disk

    # -- control
    def control(self, run_id: str) -> ControlState:
        with self.db.session() as s, s.begin():
            row = s.get(ControlRow, run_id)
            if row is None:
                st = ControlState(run_id=run_id)
                s.add(ControlRow(run_id=run_id, holder=st.holder, epoch=st.epoch, operator=st.operator,
                                 since=st.since, reason=st.reason))
                return st
            return ControlState.model_validate({"run_id": run_id, "holder": row.holder, "epoch": row.epoch,
                                                "operator": row.operator, "since": row.since, "reason": row.reason})

    def _transfer(self, s, run_id: str, holder: str, operator: str | None, reason: str) -> None:
        row = s.get(ControlRow, run_id)
        epoch = (row.epoch if row else 0) + 1
        since = _now()
        if row is None:
            s.add(ControlRow(run_id=run_id, holder=holder, epoch=epoch, operator=operator, since=since, reason=reason))
        else:
            row.holder, row.epoch, row.operator, row.since, row.reason = holder, epoch, operator, since, reason
        s.add(ControlTransferRow(run_id=run_id, holder=holder, epoch=epoch, operator=operator, since=since, reason=reason))

    def transfers(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.scalars(select(ControlTransferRow).where(ControlTransferRow.run_id == run_id)
                             .order_by(ControlTransferRow.seq))
            return [{"run_id": r.run_id, "holder": r.holder, "epoch": r.epoch, "operator": r.operator,
                     "since": r.since, "reason": r.reason} for r in rows]

    # -- requests
    def open_request(self, req: InterventionRequest) -> None:
        self.control(req.run_id)
        with self.db.session() as s, s.begin():
            s.add(InterventionRow(id=req.id, run_id=req.run_id, status=req.status, body=req.model_dump(mode="json")))

    def load(self, run_id: str, req_id: str) -> InterventionRequest:
        with self.db.session() as s:
            row = s.get(InterventionRow, req_id)
            if row is None or row.run_id != run_id:
                raise KeyError(f"no request {req_id} in run {run_id}")
            return InterventionRequest.model_validate(row.body)

    def save(self, req: InterventionRequest) -> None:
        with self.db.session() as s, s.begin():
            s.execute(update(InterventionRow).where(InterventionRow.id == req.id)
                      .values(status=req.status, body=req.model_dump(mode="json")))

    def list(self, status: str | None = "open") -> list[InterventionRequest]:
        with self.db.session() as s:
            q = select(InterventionRow).order_by(InterventionRow.updated_at)
            if status == "open":
                q = q.where(InterventionRow.status.in_(("open", "in_progress")))
            elif status:
                q = q.where(InterventionRow.status == status)
            return [InterventionRequest.model_validate(r.body) for r in s.scalars(q)]

    def claim(self, run_id: str, req_id: str, operator: str) -> InterventionRequest:
        req = self.load(run_id, req_id)
        req.status, req.claimed_by, req.claimed_at = "in_progress", operator, _now()
        with self.db.session() as s, s.begin():
            n = cast(CursorResult[Any], s.execute(update(InterventionRow)
                          .where(InterventionRow.id == req_id, InterventionRow.status == "open")
                          .values(status="in_progress", body=req.model_dump(mode="json")))).rowcount
            if n != 1:
                raise ValueError(f"request {req_id} is not open (already claimed or resolved)")
            self._transfer(s, run_id, "human", operator, f"claimed {req_id}")
        return req

    def resolve(self, run_id: str, req_id: str, decision: Decision, operator: str, note: str | None = None) -> InterventionRequest:
        req = self.load(run_id, req_id)
        if req.status == "resolved":
            raise ValueError(f"request {req_id} already resolved")
        if decision not in req.allowed_decisions and decision not in ("abort", "timeout"):
            raise ValueError(f"decision {decision} not allowed for this request ({req.allowed_decisions})")
        prev = req.status
        req.status, req.decision, req.note, req.resolved_by, req.resolved_at = "resolved", decision, note, operator, _now()
        with self.db.session() as s, s.begin():
            n = cast(CursorResult[Any], s.execute(update(InterventionRow)
                          .where(InterventionRow.id == req_id, InterventionRow.status == prev)
                          .values(status="resolved", body=req.model_dump(mode="json")))).rowcount
            if n != 1:
                raise ValueError(f"request {req_id} changed concurrently; reload and retry")
            row = s.get(ControlRow, run_id)
            if row is not None and row.holder != "automation":
                self._transfer(s, run_id, "automation", None, f"resolved {req_id}: {decision}")
        return req
