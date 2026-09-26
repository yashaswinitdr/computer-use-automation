"""Produce the /evidence set end to end:  python scripts/demo.py [--provider anthropic|openai]

Requires both target-app variants running (python -m cua app --variant a --port 5055 / --variant b --port 5057)
and an LLM key for the discovery runs. Every other step is deterministic (no LLM).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = [sys.executable, "-m", "cua"]
rows: list[dict] = []


def run(label: str, args: list[str], note: str) -> dict:
    print(f"\n### {label}: {note}", flush=True)
    p = subprocess.run(PY + args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    sys.stderr.write(p.stderr[-4000:])
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        print(p.stdout)
        raise SystemExit(f"{label}: command failed")
    run_id = out["run_id"]
    dst = OUT / label
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ROOT / "runs" / run_id, dst)
    (dst / "stdout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    fail = out.get("failure") or {}
    rows.append({"dir": label, "note": note, "status": out.get("status"),
                 "detail": (out.get("outcome") or {}).get("code") or fail.get("code") or
                 ", ".join(f"{k}={v}" for k, v in (out.get("outputs") or {}).items()) or out.get("artifact", ""),
                 "recoveries": ", ".join(r["code"] for r in out.get("recoveries", [])),
                 "interventions": len(out.get("interventions", []))})
    print(json.dumps(out, indent=2)[:1500])
    return out


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="evidence")
    a = ap.parse_args()
    OUT = ROOT / a.out
    OUT.mkdir(exist_ok=True)
    for port in (5055, 5057):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nav", timeout=3)
        except OSError:
            raise SystemExit(f"target app not running on :{port} (python -m cua app --variant {'a' if port == 5055 else 'b'} --port {port})")
    llm = (["--provider", a.provider] if a.provider else []) + (["--model", a.model] if a.model else [])
    if a.provider == "scripted":
        llm += ["--script", "scripts/scripted/savings_balance.json"]
    for p in (ROOT / "capabilities").glob("coreteller.*.json"):
        p.unlink()

    # ---- 1. LLM discovery -> artifact
    d = run("01-discovery-savings-balance", ["discover", *llm, "--no-escalate",
            "--goal", "Look up member {member_id} and read their current share savings balance",
            "--param", "member_id=10042", "--id", "coreteller.member.savings_balance",
            "--title", "Member share savings balance",
            "--description", "Looks up a member by member number and returns the current balance of their share savings account."],
            "LLM-driven discovery run (observe -> decide -> act) on the live legacy app")
    art = ROOT / d["artifact"]
    shutil.copy(art, OUT / "capability.savings_balance.json")
    review = subprocess.run(PY + ["show", str(art)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    (OUT / "capability.savings_balance.review.txt").write_text(review, encoding="utf-8")
    print(review)
    subprocess.run(PY + ["approve", str(art), "--reviewer", "demo-reviewer"], cwd=ROOT, check=True)
    cap = str(art)

    # ---- 2. deterministic replays (no LLM)
    rp = ["replay", cap, "--no-escalate"]
    run("02-replay-success", rp + ["--param", "member_id=20077"], "replay with a different member (no LLM)")
    run("03-replay-not-found", rp + ["--param", "member_id=99999"], "unknown member -> business outcome, not an error")
    run("04-replay-restricted", rp + ["--param", "member_id=30001"], "permission denial -> business outcome")
    run("05-replay-bad-input", rp + ["--param", "member_id=12ab"], "input fails the artifact's contract -> rejected before touching the UI")
    run("06-replay-recover-interstitial", rp + ["--param", "member_id=10042", "--inject", "interstitial=1", "after=1"],
        "maintenance interstitial -> recognised, acknowledged, continued")
    run("07-replay-recover-session-expiry", rp + ["--param", "member_id=10042", "--inject", "session_expire=1", "after=2"],
        "session expired mid-flow -> re-sign-on via app profile flow, restart")
    run("08-replay-slow-host", rp + ["--param", "member_id=10042", "--inject", "slow_ms=2000"], "2s host latency on every request -> waits, no false errors")
    run("09-replay-host-abend", rp + ["--param", "member_id=10042", "--inject", "abend=1", "after=2"],
        "host ABEND -> hard failure with step, expected/observed, screenshot, DOM digest")
    run("10-replay-unknown-screen-fail", rp + ["--param", "member_id=10042", "--inject", "survey=1", "after=1"],
        "screen not in the app profile, escalation disabled -> checkpoint_timeout with evidence")
    run("11-replay-unknown-screen-handoff", ["replay", cap, "--param", "member_id=10042", "--inject", "survey=1", "after=1",
        "--operator-script", "scripts/operator/stuck_skip_survey.json"],
        "same screen, escalation on -> intervention request, operator takes control of the live session, acts, hands back; run completes")
    run("12-replay-tenant-b", rp + ["--param", "member_id=10042", "--tenant", "tenants/rvfcu.json"],
        "same artifact on tenant B (same product, different captions) via tenant alias overlay")
    run("13-invoke-from-catalog", ["invoke", "coreteller__member__savings_balance", "--args", '{"member_id": "20077"}', "--no-escalate"],
        "agent-facing: invoke approved capability by tool name with JSON args")

    # ---- 3. write flow with an irreversible step
    llm2 = list(llm)
    if a.provider == "scripted":
        llm2[-1] = "scripts/scripted/open_sub_account.json"
    d2 = run("14-discovery-open-sub-account", ["discover", *llm2, "--operator-script", "scripts/operator/approve.json",
             "--goal", "Open a new {share_type} sub-account named {nickname} for member {member_id} with an initial deposit of "
                       "{initial_deposit}, and return the confirmation number and the new suffix",
             "--param", "member_id=20077", "--param", "share_type=HOLIDAY CLUB", "--param", "nickname=XMAS FUND",
             "--param", "initial_deposit=50.00", "--id", "coreteller.member.open_sub_account",
             "--title", "Open member sub-account",
             "--description", "Opens a new share sub-account for an existing member and returns the host confirmation number."],
             "LLM discovery of a multi-field write flow; the irreversible Confirm is gated by human approval")
    art2 = ROOT / d2["artifact"]
    shutil.copy(art2, OUT / "capability.open_sub_account.json")
    (OUT / "capability.open_sub_account.review.txt").write_text(
        subprocess.run(PY + ["show", str(art2)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout, encoding="utf-8")
    subprocess.run(PY + ["approve", str(art2), "--reviewer", "demo-reviewer"], cwd=ROOT, check=True)
    sub_args = lambda d: json.dumps({"member_id": "10042", "share_type": "VACATION CLUB", "nickname": "BEACH", "initial_deposit": d})
    run("15-replay-sub-account-approved", ["replay", str(art2), "--args", sub_args("75.00"), "--operator-script", "scripts/operator/approve.json"],
        "irreversible step -> approval request -> approved -> committed")
    run("16-replay-sub-account-denied", ["replay", str(art2), "--args", sub_args("75.00"), "--operator-script", "scripts/operator/deny.json"],
        "irreversible step -> approval denied -> stops, nothing committed")
    run("17-replay-sub-account-host-validation", ["replay", str(art2), "--args", sub_args("9999"), "--no-escalate"],
        "host validation error (ERR 1102) before the irreversible step -> business outcome invalid_input")

    lines = ["# Evidence index", "", "Generated by `python scripts/demo.py`. Each directory is a copy of `runs/<run_id>/`: "
             "`events.jsonl` (structured, redacted log), `result.json`/`stdout.json`, `screenshots/` (PII-masked), "
             "`dom/` (redacted text + control inventory, on failure/escalation), `handoff/` (intervention requests, control log).",
             "", "| dir | what it shows | status | outputs / outcome / failure | recoveries | interventions |", "|---|---|---|---|---|---|"]
    lines += [f"| `{r['dir']}` | {r['note']} | {r['status']} | {r['detail']} | {r['recoveries']} | {r['interventions']} |" for r in rows]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
