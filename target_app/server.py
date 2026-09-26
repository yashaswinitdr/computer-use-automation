"""Acme CoreTeller 4.2 -- a deliberately "legacy" back-office banking app used as the proxy target.

Everything here is synthetic. It imitates the properties that make real core-banking screens hard to automate:

* a <frameset> shell (nav frame + main frame), so the interesting content never lives in the top document;
* table-based layout with labels in sibling <td> cells (no <label for>, no ids, no test ids, no ARIA);
* opaque field names (f1, f2, q, x9), buttons that submit via inline javascript;
* status messages rendered as red <font> text rather than role=alert;
* runtime conditions: record-not-found, restricted-account denial, validation errors, a maintenance
  interstitial, session expiry, host "ABEND" errors, and slow responses.

Faults are injected through /__faults (a test-harness endpoint that the automation policy does NOT allowlist).
TENANT_VARIANT=b serves the same product "configured differently" (different field captions), standing in for
a second institution running the same vendor product.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass, field

from flask import Flask, Response, redirect, request, session

app = Flask(__name__)
app.secret_key = os.environ.get("TARGET_APP_SECRET", "dev-only-not-a-secret")

# Seed data. Test credentials live in seed.py so the automation's example config can reference them.
from target_app.seed import MEMBERS, SESSION_TTL_SECONDS, TELLERS  # noqa: E402

VARIANT = os.environ.get("TENANT_VARIANT", "a").lower()
LABELS = {
    "a": {"member_no": "Member Number:", "search": "Search", "balance": "Balance", "brand": "FIRST COMMUNITY CU"},
    "b": {"member_no": "Acct/Member #:", "search": "Find", "balance": "Current Bal", "brand": "RIVER VALLEY FCU"},
}[VARIANT]


# ---------------------------------------------------------------- fault injection (test harness only)
@dataclass
class Faults:
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending: dict[str, int] = field(default_factory=dict)  # fault name -> remaining count
    slow_ms: int = 0
    after: int = 0  # let this many authenticated requests pass before any fault fires

    def arm(self) -> bool:
        """Called once per authenticated request: True once `after` requests have gone by."""
        with self.lock:
            if self.after > 0:
                self.after -= 1
                return False
            return True

    def take(self, name: str, armed: bool = True) -> bool:
        with self.lock:
            if not armed:
                return False
            n = self.pending.get(name, 0)
            if n <= 0:
                return False
            self.pending[name] = n - 1
            return True


FAULTS = Faults()
_sub_accounts: dict[str, list[dict]] = {}


@app.route("/__faults", methods=["POST", "GET"])
def set_faults():
    """?interstitial=1&session_expire=1&abend=1&slow_ms=2500&after=3 ; ?reset=1 clears everything."""
    with FAULTS.lock:
        if request.args.get("reset"):
            FAULTS.pending.clear()
            FAULTS.slow_ms = 0
            FAULTS.after = 0
            _sub_accounts.clear()
        for name in ("interstitial", "session_expire", "abend", "survey"):
            if name in request.args:
                FAULTS.pending[name] = int(request.args[name])
        if "after" in request.args:
            FAULTS.after = int(request.args["after"])
        if "slow_ms" in request.args:
            FAULTS.slow_ms = int(request.args["slow_ms"])
        state = {"pending": dict(FAULTS.pending), "slow_ms": FAULTS.slow_ms, "variant": VARIANT}
    return state


# ---------------------------------------------------------------- page chrome
def page(title: str, body: str, status: int = 200) -> Response:
    html = f"""<html><head><title>{title}</title>
<style>body{{font-family:Courier New,monospace;font-size:13px;background:#c0c0c0}}
td{{font-size:13px}} .hdr{{background:#000080;color:#fff;font-weight:bold}}</style></head>
<body><table width="100%" cellpadding="2" cellspacing="0"><tr><td class="hdr">{LABELS['brand']} &nbsp;-&nbsp; CORETELLER 4.2</td>
<td class="hdr" align="right">{time.strftime('%m/%d/%Y')}</td></tr></table><br>{body}</body></html>"""
    return Response(html, status=status)


def error_text(msg: str) -> str:
    return f'<font color="#cc0000"><b>{msg}</b></font><br><br>'


@app.before_request
def runtime_conditions():
    if request.path.startswith("/__") or request.path in ("/", "/nav", "/favicon.ico"):
        return None
    if FAULTS.slow_ms:
        time.sleep(FAULTS.slow_ms / 1000)
    if request.path in ("/login", "/notice/ack"):
        return None
    # authenticated area
    if not session.get("teller"):
        return redirect("/login")
    armed = FAULTS.arm()
    if time.time() - session.get("seen", 0) > SESSION_TTL_SECONDS or FAULTS.take("session_expire", armed):
        session.clear()
        return page("SIGN ON", error_text("SESSION EXPIRED - PLEASE SIGN ON AGAIN") + login_form(), 200)
    session["seen"] = time.time()
    if FAULTS.take("abend", armed):
        return page("SYSTEM ERROR", error_text("ABEND S0C7 - TRANSACTION TERMINATED. CONTACT HELP DESK. REF " + secrets.token_hex(3).upper()), 500)
    if request.method == "GET" and FAULTS.take("interstitial", armed):
        session["after_notice"] = request.full_path
        return page("SYSTEM NOTICE", """
<table border="1" cellpadding="8" bgcolor="#ffffcc"><tr><td>
<b>SYSTEM NOTICE</b><br><br>SCHEDULED HOST MAINTENANCE TONIGHT 11:00 PM - 1:00 AM ET.<br>
SAVE ALL WORK BEFORE 10:55 PM.<br><br>
<form method="post" action="/notice/ack"><input type="submit" value="Acknowledge"></form>
</td></tr></table>""")
    if request.method == "GET" and FAULTS.take("survey", armed):
        # A screen the automation has never seen (not in the app profile): used to demonstrate human handoff.
        session["after_notice"] = request.full_path
        return page("TELLER SURVEY", """
<b>QUICK TELLER SURVEY</b><br><br>HOW SATISFIED ARE YOU WITH CORETELLER 4.2?<br><br>
<form method="post" action="/notice/ack"><select name="s"><option>VERY SATISFIED</option><option>SATISFIED</option>
<option>NOT SATISFIED</option></select><br><br><input type="submit" value="Submit Survey"></form>
<form method="post" action="/notice/ack"><input type="submit" value="Remind Me Later"></form>""")
    return None


@app.route("/notice/ack", methods=["POST"])
def notice_ack():
    return redirect(session.pop("after_notice", "/home"))


# ---------------------------------------------------------------- shell
@app.route("/")
def shell():
    return """<html><head><title>CoreTeller</title></head>
<frameset rows="60,*" border="1">
  <frame name="nav" src="/nav" scrolling="no">
  <frame name="main" src="/home">
</frameset></html>"""


@app.route("/nav")
def nav():
    return Response("""<html><head><title>MENU</title></head><body bgcolor="#000080">
<table cellpadding="4"><tr>
<td><a href="/home" target="main"><font color="white">Home</font></a></td><td><font color="white">|</font></td>
<td><a href="/inq" target="main"><font color="white">Member Inquiry</font></a></td><td><font color="white">|</font></td>
<td><a href="/logout" target="main"><font color="white">Sign Off</font></a></td>
</tr></table></body></html>""")


def login_form() -> str:
    return """<form method="post" action="/login"><table>
<tr><td>User ID:</td><td><input type="text" name="f1" size="10"></td></tr>
<tr><td>Password:</td><td><input type="password" name="f2" size="10"></td></tr>
<tr><td></td><td><input type="submit" value="Sign On"></td></tr></table></form>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user, pw = request.form.get("f1", "").upper(), request.form.get("f2", "")
        if TELLERS.get(user) == pw:
            session.update(teller=user, seen=time.time())
            return redirect("/home")
        return page("SIGN ON", error_text("INVALID USER ID OR PASSWORD") + login_form())
    return page("SIGN ON", login_form())


@app.route("/logout")
def logout():
    session.clear()
    return page("SIGN ON", "SIGNED OFF.<br><br>" + login_form())


@app.route("/home")
def home():
    return page("MAIN MENU", f"WELCOME, {session['teller']}.<br><br>SELECT A FUNCTION FROM THE MENU ABOVE.")


# ---------------------------------------------------------------- member inquiry
def inquiry_form(value: str = "") -> str:
    return f"""<form method="get" action="/inq"><table>
<tr><td>{LABELS['member_no']}</td><td><input type="text" name="q" size="12" value="{value}"></td>
<td><input type="button" value="{LABELS['search']}" onclick="document.forms[0].submit()"></td></tr></table></form>"""


@app.route("/inq")
def inquiry():
    q = request.args.get("q", "").strip()
    if not q:
        return page("MEMBER INQUIRY", inquiry_form())
    if not q.isdigit():
        return page("MEMBER INQUIRY", error_text("ERR 0107: MEMBER NUMBER MUST BE NUMERIC") + inquiry_form(q))
    m = MEMBERS.get(q)
    if m is None:
        return page("MEMBER INQUIRY", error_text(f"NO RECORD FOUND FOR MEMBER {q}") + inquiry_form(q))
    if m.get("restricted"):
        return page("MEMBER INQUIRY", error_text("ACCESS DENIED - RESTRICTED ACCOUNT (SEC-403). REFER TO BSA OFFICER.") + inquiry_form(q))
    return redirect(f"/mbr/{q}")


@app.route("/mbr/<mid>")
def member(mid: str):
    m = MEMBERS.get(mid)
    if m is None or m.get("restricted"):
        return redirect(f"/inq?q={mid}")
    rows = "".join(
        f"<tr><td>{s['suffix']}</td><td>{s['type']}</td><td>{s['desc']}</td><td align='right'>${s['bal']:,.2f}</td></tr>"
        for s in m["shares"] + _sub_accounts.get(mid, [])
    )
    return page("MEMBER SUMMARY", f"""
<table border="0" cellpadding="3">
<tr><td><b>Member:</b></td><td>{mid}</td><td width="40"></td><td><b>Name:</b></td><td>{m['name']}</td></tr>
<tr><td><b>SSN:</b></td><td>{m['ssn']}</td><td></td><td><b>Member Since:</b></td><td>{m['since']}</td></tr>
</table><br>
<table border="1" cellpadding="3" cellspacing="0">
<tr bgcolor="#e0e0e0"><td><b>Suffix</b></td><td><b>Type</b></td><td><b>Description</b></td><td><b>{LABELS['balance']}</b></td></tr>
{rows}
</table><br>
<a href="/sub/new?m={mid}">Open Sub-Account</a> &nbsp; <a href="/inq">New Inquiry</a>""")


# ---------------------------------------------------------------- open sub-account (multi-step, irreversible confirm)
SHARE_TYPES = {"01": "SAVINGS", "05": "HOLIDAY CLUB", "06": "VACATION CLUB", "09": "MONEY MARKET"}


def sub_form(mid: str, err: str = "", vals: dict | None = None) -> Response:
    vals = vals or {}
    opts = "".join(
        f'<option value="{k}"{" selected" if vals.get("x1") == k else ""}>{v}</option>' for k, v in SHARE_TYPES.items()
    )
    return page("OPEN SUB-ACCOUNT", f"""{error_text(err) if err else ''}
<form method="post" action="/sub/review"><input type="hidden" name="m" value="{mid}"><table>
<tr><td>Member:</td><td>{mid}</td></tr>
<tr><td>Share Type:</td><td><select name="x1">{opts}</select></td></tr>
<tr><td>Nickname:</td><td><input type="text" name="x2" size="20" value="{vals.get('x2', '')}"></td></tr>
<tr><td>Initial Deposit:</td><td><input type="text" name="x9" size="10" value="{vals.get('x9', '')}"></td></tr>
<tr><td></td><td><input type="submit" value="Continue"></td></tr></table></form>""")


@app.route("/sub/new")
def sub_new():
    mid = request.args.get("m", "")
    if mid not in MEMBERS:
        return redirect("/inq")
    return sub_form(mid)


@app.route("/sub/review", methods=["POST"])
def sub_review():
    f = request.form
    mid = f.get("m", "")
    if not f.get("x2", "").strip():
        return sub_form(mid, "ERR 1101: NICKNAME IS REQUIRED", f)
    try:
        amt = float(f.get("x9", "").replace("$", "").replace(",", ""))
        if not 0 <= amt <= 5000:
            raise ValueError
    except ValueError:
        return sub_form(mid, "ERR 1102: INVALID DEPOSIT AMOUNT (0.00 - 5,000.00)", f)
    session["pending_sub"] = {"m": mid, "type": f["x1"], "nick": f["x2"].strip().upper(), "amt": amt}
    p = session["pending_sub"]
    return page("REVIEW SUB-ACCOUNT", f"""
<b>PLEASE REVIEW. THIS ACTION CANNOT BE UNDONE.</b><br><br><table cellpadding="3">
<tr><td>Member:</td><td>{mid}</td></tr><tr><td>Share Type:</td><td>{SHARE_TYPES[p['type']]}</td></tr>
<tr><td>Nickname:</td><td>{p['nick']}</td></tr><tr><td>Initial Deposit:</td><td>${amt:,.2f}</td></tr></table><br>
<form method="post" action="/sub/confirm"><input type="submit" value="Confirm"></form>
<form method="get" action="/mbr/{mid}"><input type="submit" value="Cancel"></form>""")


@app.route("/sub/confirm", methods=["POST"])
def sub_confirm():
    p = session.pop("pending_sub", None)
    if not p:
        return page("OPEN SUB-ACCOUNT", error_text("ERR 1199: NO PENDING REQUEST"))
    subs = _sub_accounts.setdefault(p["m"], [])
    suffix = f"{20 + len(subs):02d}"
    subs.append({"suffix": suffix, "type": p["type"], "desc": p["nick"], "bal": p["amt"]})
    conf = secrets.token_hex(3).upper()
    return page("SUB-ACCOUNT OPENED", f"""<b>SUB-ACCOUNT {suffix} OPENED.</b><br><br><table cellpadding="3">
<tr><td>Confirmation #:</td><td>{conf}</td></tr><tr><td>Suffix:</td><td>{suffix}</td></tr></table><br>
<a href="/mbr/{p['m']}">Return to Member</a>""")


def main() -> None:
    port = int(os.environ.get("TARGET_APP_PORT", "5055"))
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
