"""Command line entry point:  python -m cua <command> ...   (see README.md for the demo path)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from cua import catalog as cat
from cua.evidence import RunLog, new_run_id
from cua.handoff import Controller, HandoffStore, load_operator_script
from cua.policy import Policy
from cua.redact import Redactor
from cua.schema import Capability

ROOT = cat.ROOT
RUNS = ROOT / "runs"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


def parse_params(items: list[str] | None) -> dict[str, str]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = v
    return out


def inject_faults(base_url: str, spec: list[str] | None) -> None:
    q = {"reset": "1"}
    for s in spec or []:
        k, _, v = s.partition("=")
        q[k] = v or "1"
    urllib.request.urlopen(f"{base_url}/__faults?{urllib.parse.urlencode(q)}", timeout=5).read()


def _console(args) -> None:
    if args.no_escalate or getattr(args, "operator_script", None):
        return
    from cua.operator_console import serve

    try:
        serve(HandoffStore(RUNS), port=args.console_port, background=True)
        print(f"operator console: http://127.0.0.1:{args.console_port}/", file=sys.stderr)
    except OSError:
        print(f"operator console already running on :{args.console_port}", file=sys.stderr)


def _setup(kind: str, args):
    from cua.surface.web import WebSurface

    tenant = cat.load_tenant(args.tenant)
    profile = cat.load_profile(tenant.app_profile)
    policy = Policy.load(args.policy)
    redactor = Redactor()
    run_id = new_run_id(kind)
    log = RunLog(RUNS, run_id, redactor)
    surface = WebSurface(tenant.base_url, policy, headless=not args.headed, slow_mo=args.slow_mo)
    ctl = Controller(HandoffStore(RUNS), run_id, log, surface, operator=load_operator_script(getattr(args, "operator_script", None)),
                     timeout_s=args.handoff_timeout, pii_labels=profile.pii_labels)
    ctl.enabled = not args.no_escalate
    return tenant, profile, policy, log, surface, ctl


# ------------------------------------------------------------------------------------------------ commands
def cmd_discover(args) -> int:
    from cua.agent.discovery import Discovery
    from cua.agent.llm import make_llm
    from cua.recorder import build_capability
    from cua.replay import Replayer

    params = parse_params(args.param)
    goal = args.goal.format(**params)
    llm = make_llm(args.provider, args.model, args.script)
    _console(args)
    tenant, profile, policy, log, surface, ctl = _setup("discover", args)
    if args.inject:
        inject_faults(tenant.base_url, args.inject)
    try:
        session = Replayer(surface, profile, tenant, policy, log, ctl)
        d = Discovery(surface, llm, policy, log, ctl, session, profile.pii_labels, max_steps=args.max_steps)
        outcome = d.run(goal, params)
        log.write_json("trace.json", [t.__dict__ for t in outcome.trace])
        if outcome.status != "succeeded":
            print(json.dumps({"status": "failed", "run_id": log.run_id, "reason": outcome.reason}, indent=2))
            return 2
        cap = build_capability(outcome, capability_id=args.id, title=args.title or args.goal.rstrip("."),
                               description=args.description or f"Recorded from goal: {args.goal}",
                               goal_template=args.goal, params=params, profile=profile, run_id=log.run_id, model=llm.model)
        path = cat.capability_path(cap)
        while path.exists():
            major, minor, _ = map(int, cap.version.split("."))
            cap.version = f"{major}.{minor + 1}.0"
            path = cat.capability_path(cap)
        path.write_text(cap.to_json(), encoding="utf-8")
        (log.dir / "artifact.json").write_text(cap.to_json(), encoding="utf-8")
        log.event("artifact_written", path=str(path.relative_to(ROOT)), capability=cap.ref, steps=len(cap.steps))
        print(json.dumps({"status": "succeeded", "run_id": log.run_id, "artifact": str(path.relative_to(ROOT)),
                          "outputs_seen_during_discovery": sorted(outcome.outputs)}, indent=2))
        return 0
    finally:
        surface.close()
        log.close()


def cmd_replay(args) -> int:
    from cua.replay import Replayer

    cap = cat.load_capability(args.capability)
    _console(args)
    tenant, profile, policy, log, surface, ctl = _setup("replay", args)
    try:
        if args.inject is not None:
            inject_faults(tenant.base_url, args.inject)
        res = Replayer(surface, profile, tenant, policy, log, ctl, escalate=not args.no_escalate).run(
            cap, parse_params(args.param) | (json.loads(args.args) if args.args else {}), allow_draft=args.allow_draft)
    finally:
        surface.close()
        log.close()
    out = res.model_dump(mode="json", exclude_none=True)
    if not args.reveal:
        out["outputs"] = json.loads((RUNS / res.run_id / "result.json").read_text())["outputs"]
    print(json.dumps(out, indent=2))
    return {"succeeded": 0, "business_outcome": 3}.get(res.status, 4)


def cmd_approve(args) -> int:
    p = Path(args.artifact)
    cap = Capability.model_validate_json(p.read_text(encoding="utf-8"))
    cap.status = "approved"
    cap.provenance.reviewed_by = args.reviewer
    cap.provenance.reviewed_at = datetime.now(timezone.utc)
    p.write_text(cap.to_json(), encoding="utf-8")
    print(f"{cap.ref} approved by {args.reviewer}")
    return 0


def cmd_show(args) -> int:
    cap = cat.load_capability(args.capability)
    print(f"{cap.ref}  [{cap.status}]  risk={cap.risk}  app={cap.app.profile}@{cap.app.profile_version}")
    print(f"  {cap.title}\n  {cap.description}")
    print("  inputs:  " + (", ".join(f"{n}:{s.type}({s.sensitivity.value})" for n, s in cap.inputs.items()) or "-"))
    print("  outputs: " + (", ".join(f"{n}:{o.type}({o.sensitivity.value})" for n, o in cap.outputs.items()) or "-"))
    print("  business outcomes: " + ", ".join(o.code for o in cap.outcomes))
    for s in cap.steps:
        a = s.action
        what = a.target.describe() if hasattr(a, "target") else getattr(a, "output", getattr(a, "route", ""))
        val = getattr(a, "value", None)
        v = f" <- {{{val.name}}}" if val is not None and val.kind == "param" else (f" <- {val.value!r}" if val is not None and val.kind == "literal" else "")
        exp = " ; ".join(f"{c.kind}={getattr(c, 'pattern', None) or getattr(c, 'equals', None) or getattr(c, 'contains', None)}"
                         for c in (s.expect.all_of if s.expect else []))
        print(f"  {s.id:>4} [{s.risk:12}] {a.type:8} {what}{v}" + (f"   => {exp}" if exp else "") + (f"   ({s.source})" if s.source != "model" else ""))
    print("  success: " + " ; ".join(c.model_dump_json(exclude_none=True) for c in cap.success.all_of))
    return 0


def cmd_catalog(args) -> int:
    print(cat.dumps(cat.catalog(include_drafts=args.include_drafts)))
    return 0


def cmd_invoke(args) -> int:
    """Agent-facing entry point: call an approved capability by tool name with JSON args."""
    cid = args.name.replace("__", ".")
    args.capability = cat.capability_path(cat.load_capability(cid, approved_only=True)).as_posix()
    args.param, args.allow_draft = [], False
    return cmd_replay(args)


def cmd_operator(args) -> int:
    store = HandoffStore(RUNS)
    if args.op == "console":
        from cua.operator_console import serve

        print(f"operator console on http://127.0.0.1:{args.port}/")
        serve(store, port=args.port, operator_name=args.as_)
    elif args.op == "list":
        for r in store.list(status=None if args.all else "open"):
            print(f"{r.run_id}  {r.id}  {r.kind:13} {r.status:11} {r.reason[:90]}")
    elif args.op == "claim":
        store.claim(args.run, args.req, args.as_)
        print(f"control of {args.run} -> human ({args.as_})")
    elif args.op == "resolve":
        store.resolve(args.run, args.req, args.decision, args.as_, args.note)
        print(f"{args.req} resolved: {args.decision}")
    return 0


def cmd_app(args) -> int:
    os.environ["TENANT_VARIANT"] = args.variant
    os.environ["TARGET_APP_PORT"] = str(args.port)
    from target_app import server  # reads TENANT_VARIANT at import

    server.main()
    return 0


def evidence_copy(args) -> int:
    """Copy a run directory into evidence/<name>/ (runs/ is git-ignored)."""
    dst = ROOT / "evidence" / args.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(RUNS / args.run_id, dst)
    print(f"copied {args.run_id} -> {dst.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="cua", description="Computer-use automation: discover, record, replay")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, escalate=True):
        p.add_argument("--tenant", default=str(ROOT / "tenants" / "fccu.json"))
        p.add_argument("--policy", default=str(ROOT / "policy.json"))
        p.add_argument("--headed", action="store_true", help="show the browser (needed for a human to take control)")
        p.add_argument("--slow-mo", type=int, default=0)
        p.add_argument("--inject", nargs="*", help="fault injection on the test target, e.g. interstitial session_expire abend slow_ms=2500")
        p.add_argument("--no-escalate", action="store_true", help="never wait for a human; fail instead")
        p.add_argument("--operator-script", help="JSON script for the simulated operator (demo/test stand-in for a human)")
        p.add_argument("--handoff-timeout", type=float, default=600)
        p.add_argument("--console-port", type=int, default=5056)

    p = sub.add_parser("discover", help="LLM-driven run on a goal; records a capability artifact on success")
    p.add_argument("--goal", required=True, help="goal template, may reference {param}s")
    p.add_argument("--param", action="append", help="name=value (example value used for this run)")
    p.add_argument("--id", required=True, help="capability id, e.g. coreteller.member.savings_balance")
    p.add_argument("--title")
    p.add_argument("--description")
    p.add_argument("--provider", choices=["anthropic", "openai", "ollama", "scripted"])
    p.add_argument("--model")
    p.add_argument("--script", help="scripted provider: JSON list of tool calls (offline tests only)")
    p.add_argument("--max-steps", type=int, default=20)
    common(p)
    p.set_defaults(fn=cmd_discover)

    p = sub.add_parser("replay", help="deterministic replay of a capability (no LLM)")
    p.add_argument("capability", help="artifact path or id[@version]")
    p.add_argument("--param", action="append")
    p.add_argument("--args", help="inputs as a JSON object")
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--reveal", action="store_true", help="print pii outputs unmasked (caller view)")
    common(p)
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("invoke", help="agent-facing: invoke an APPROVED capability by tool name")
    p.add_argument("name")
    p.add_argument("--args", required=True)
    p.add_argument("--reveal", action="store_true")
    common(p)
    p.set_defaults(fn=cmd_invoke)

    p = sub.add_parser("approve", help="mark a reviewed artifact approved")
    p.add_argument("artifact")
    p.add_argument("--reviewer", required=True)
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("show", help="human-readable review of an artifact")
    p.add_argument("capability")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("catalog", help="approved capabilities as agent tool definitions")
    p.add_argument("--include-drafts", action="store_true")
    p.set_defaults(fn=cmd_catalog)

    p = sub.add_parser("operator", help="operator side of the handoff")
    p.add_argument("op", choices=["console", "list", "claim", "resolve"])
    p.add_argument("run", nargs="?")
    p.add_argument("req", nargs="?")
    p.add_argument("decision", nargs="?", choices=["approve", "deny", "resume", "abort"])
    p.add_argument("--as", dest="as_", default=os.environ.get("USERNAME", "operator"))
    p.add_argument("--note")
    p.add_argument("--port", type=int, default=5056)
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_operator)

    p = sub.add_parser("app", help="run the synthetic legacy target app")
    p.add_argument("--variant", default="a", choices=["a", "b"])
    p.add_argument("--port", type=int, default=5055)
    p.set_defaults(fn=cmd_app)

    p = sub.add_parser("keep", help="copy a run into evidence/<name>")
    p.add_argument("run_id")
    p.add_argument("name")
    p.set_defaults(fn=evidence_copy)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
