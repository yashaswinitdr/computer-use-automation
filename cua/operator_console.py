"""Minimal operator console over the handoff store (deliberately bare -- the seam is what matters).

Shows open intervention requests with their reason, step, redacted state and masked screenshot, and lets an
operator Take control / Approve / Deny / Resume / Abort. The live session itself is the automation's browser
window (run with --headed); a production console would stream it (CDP screencast / VNC), see REPORT.md.
"""

from __future__ import annotations

import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from cua.handoff import HandoffStore

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Operator console</title>
<meta http-equiv="refresh" content="3"><style>
body{{font:14px system-ui,sans-serif;margin:16px;max-width:1100px}} .req{{border:1px solid #999;border-radius:6px;padding:12px;margin:12px 0}}
.k{{color:#555}} pre{{background:#f4f4f4;padding:8px;white-space:pre-wrap;max-height:180px;overflow:auto}}
img{{max-width:100%;border:1px solid #ccc}} button{{margin-right:6px;padding:6px 12px}} .ctl{{font-weight:bold}}
</style></head><body><h2>Intervention requests</h2>{body}</body></html>"""


def _render(store: HandoffStore) -> str:
    reqs = store.list(status="open")
    if not reqs:
        return PAGE.format(body="<p>No open requests.</p>")
    parts = []
    for r in reqs:
        ctl = store.control(r.run_id)
        esc = html.escape
        state = "\n".join(f"[{s.get('frame')}] {s.get('title')} {s.get('path')}\n{s.get('text', '')[:300]}" for s in r.state)
        buttons = []
        if r.status == "open":
            buttons.append(("claim", "Take control"))
        buttons += [(d, d.capitalize()) for d in r.allowed_decisions]
        buttons.append(("abort", "Abort run"))
        forms = "".join(
            f'<form method="post" action="/act" style="display:inline"><input type="hidden" name="run" value="{esc(r.run_id)}">'
            f'<input type="hidden" name="req" value="{esc(r.id)}"><input type="hidden" name="do" value="{d}">'
            f'<button>{label}</button></form>' for d, label in buttons)
        parts.append(f"""<div class="req"><div><b>{esc(r.kind.upper())}</b> &middot; {esc(r.id)} &middot; run {esc(r.run_id)}</div>
<div class="k">subject: {esc(r.subject)}</div><div class="k">step: {esc(str(r.step_id))} {esc(r.step_intent or '')}</div>
<p>{esc(r.reason)}</p>{'<p>proposed action: <code>' + esc(r.proposed_action) + '</code></p>' if r.proposed_action else ''}
<p class="ctl">Control: {esc(ctl.holder)}{(' (' + esc(ctl.operator) + ')') if ctl.operator else ''} &middot; epoch {ctl.epoch}</p>
<pre>{esc(state)}</pre><img src="/shot?run={esc(r.run_id)}&req={esc(r.id)}"><p>{forms}</p>
<p class="k">After "Take control", operate the live browser window, then press Resume to hand control back.</p></div>""")
    return PAGE.format(body="".join(parts))


def make_handler(store: HandoffStore, operator_name: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlsplit(self.path)
            if u.path == "/":
                return self._send(200, _render(store).encode())
            if u.path == "/shot":
                q = parse_qs(u.query)
                req = store.load(q["run"][0], q["req"][0])
                p = Path(store.root) / req.run_id / (req.screenshot or "")
                if p.is_file():
                    return self._send(200, p.read_bytes(), "image/png")
            self._send(404, b"not found")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            q = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode()).items()}
            try:
                if q["do"] == "claim":
                    store.claim(q["run"], q["req"], operator_name)
                else:
                    store.resolve(q["run"], q["req"], q["do"], operator_name)
            except (ValueError, KeyError) as e:
                return self._send(400, html.escape(str(e)).encode())
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    return H


def serve(store: HandoffStore, port: int = 5056, operator_name: str = "operator", background: bool = False):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, operator_name))
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    srv.serve_forever()
