"""Playwright-backed web surface, built for legacy markup (framesets, table layouts, no ids/test ids/ARIA).

Perception is a single JS library (``CUA_LIB``) that computes (role, accessible name, nth) for every visible
control in every frame. The *same* library is used by (a) the observation shown to the LLM during discovery,
(b) target resolution during deterministic replay, and (c) capture of human operator actions during handoff --
so what the model saw, what the artifact records and what replay matches are guaranteed to agree.
"""

from __future__ import annotations

import fnmatch
import json
import re
from typing import Any, Callable
from urllib.parse import urlsplit

from playwright.sync_api import Frame, Page, sync_playwright
from playwright.sync_api import Error as PWError

from cua.policy import Policy
from cua.schema import Condition, ElementCond, LabeledValue, RouteCond, TableCell, Target, TextCond, TextRegex, TitleCond
from cua.surface.base import FrameState, Observation, Resolved, TargetError, UIElement

CUA_LIB = r"""
const L = (() => {
  const clean = t => (t || '').replace(/\s+/g, ' ').trim();
  const caption = t => clean(t).replace(/[:\s]+$/, '');
  const norm = t => caption(t).toLowerCase();
  const SEL = 'a[href],button,input,select,textarea,[onclick],[role=button],[role=link],[role=textbox],[role=combobox],[role=checkbox]';
  const visible = el => {
    if (!el.getClientRects().length) return false;
    const s = getComputedStyle(el); return s.visibility !== 'hidden' && s.display !== 'none';
  };
  const role = el => {
    const r = el.getAttribute('role'); if (r) return r;
    const t = el.tagName.toLowerCase();
    if (t === 'a') return 'link';
    if (t === 'button') return 'button';
    if (t === 'select') return 'combobox';
    if (t === 'textarea') return 'textbox';
    if (t === 'input') {
      const ty = (el.type || 'text').toLowerCase();
      if (['submit', 'button', 'reset', 'image'].includes(ty)) return 'button';
      if (ty === 'checkbox' || ty === 'radio') return ty;
      if (ty === 'hidden') return null;
      return 'textbox';
    }
    if (el.hasAttribute('onclick')) return 'button';
    return null;
  };
  // Legacy apps put the caption in a sibling table cell: <td>Member Number:</td><td><input name=q></td>
  const cellCaption = el => {
    const td = el.closest('td, th'); if (!td) return '';
    for (let p = td.previousElementSibling; p; p = p.previousElementSibling) {
      const t = caption(p.innerText); if (t) return t;
    }
    return '';
  };
  const name = el => {
    const al = el.getAttribute('aria-label'); if (al) return caption(al);
    const lb = el.getAttribute('aria-labelledby');
    if (lb) { const n = lb.split(/\s+/).map(i => el.ownerDocument.getElementById(i)).filter(Boolean).map(e => e.innerText).join(' '); if (clean(n)) return caption(n); }
    if (el.id) { const l = el.ownerDocument.querySelector('label[for="' + CSS.escape(el.id) + '"]'); if (l) return caption(l.innerText); }
    const wl = el.closest('label'); if (wl && clean(wl.innerText)) return caption(wl.innerText);
    const t = el.tagName.toLowerCase(), ty = (el.type || '').toLowerCase();
    if (t === 'input' && ['submit', 'button', 'reset'].includes(ty)) return caption(el.value);
    if (t === 'input' && ty === 'image') return caption(el.alt);
    if (t === 'a' || t === 'button' || el.hasAttribute('onclick')) { const x = caption(el.innerText); if (x) return x; }
    const c = cellCaption(el); if (c) return c;
    return caption(el.placeholder || el.title || '');
  };
  const css = el => {
    const t = el.tagName.toLowerCase(), q = v => String(v).replace(/"/g, '\\"');
    if (t === 'input' && ['submit', 'button', 'reset'].includes((el.type || '').toLowerCase()))
      return `input[type="${el.type}"][value="${q(el.value)}"]`;
    const nm = el.getAttribute('name'); if (nm) return `${t}[name="${q(nm)}"]`;
    const parts = [];
    for (let e = el; e && e.nodeType === 1 && e.tagName !== 'BODY'; e = e.parentElement) {
      let i = 1; for (let s = e.previousElementSibling; s; s = s.previousElementSibling) if (s.tagName === e.tagName) i++;
      parts.unshift(`${e.tagName.toLowerCase()}:nth-of-type(${i})`);
    }
    return 'body > ' + parts.join(' > ');
  };
  const controls = () => {
    const out = [], counts = {};
    for (const el of document.querySelectorAll(SEL)) {
      const r = role(el); if (!r || !visible(el)) continue;
      const n = name(el), key = r + '\u0000' + n.toLowerCase();
      const nth = counts[key] || 0; counts[key] = nth + 1;
      out.push({el, role: r, name: n, nth});
    }
    return out;
  };
  const nthOf = el => { const c = controls().find(c => c.el === el); return c ? c.nth : 0; };
  return {clean, caption, norm, role, name, css, controls, nthOf, SEL};
})();
"""

SNAPSHOT_JS = "(args) => {" + CUA_LIB + r"""
  const els = !args.withElements ? [] : L.controls().map((c, i) => {
    const ref = args.prefix + i, t = c.el.tagName.toLowerCase();
    c.el.setAttribute('data-cua-ref', ref);
    const secret = t === 'input' && (c.el.type || '').toLowerCase() === 'password';
    let value = null;
    if (c.role === 'textbox' && !secret) value = c.el.value;
    if (t === 'select') value = c.el.selectedIndex >= 0 ? L.clean(c.el.options[c.el.selectedIndex].text) : null;
    return {ref, role: c.role, name: c.name, nth: c.nth, tag: t, value, secret, css: L.css(c.el),
            options: t === 'select' ? [...c.el.options].map(o => L.clean(o.text)) : []};
  });
  const body = document.body;
  return {title: document.title, path: location.pathname, text: body ? body.innerText : '', elements: els};
}"""

RESOLVE_JS = "(args) => {" + CUA_LIB + r"""
  const want = args.names.map(L.norm);
  const m = L.controls().filter(c => c.role === args.role &&
      (args.contains ? want.some(w => L.norm(c.name).includes(w)) : want.includes(L.norm(c.name))));
  return m.map((c, i) => { const ref = args.prefix + i; c.el.setAttribute('data-cua-ref', ref);
                           return {ref, name: c.name, nth: c.nth, tag: c.el.tagName.toLowerCase(), css: L.css(c.el)}; });
}"""

LABELED_VALUE_JS = "(args) => {" + CUA_LIB + r"""
  const want = args.labels.map(L.norm);
  for (const td of document.querySelectorAll('td, th')) {
    if (td.querySelector('table') || !want.includes(L.norm(td.innerText))) continue;
    for (let n = td.nextElementSibling; n; n = n.nextElementSibling) { const t = L.clean(n.innerText); if (t) return t; }
  }
  return null;
}"""

TABLE_CELL_JS = "(args) => {" + CUA_LIB + r"""
  const keyCols = args.keyCols.map(L.norm), cols = args.cols.map(L.norm), key = L.norm(args.key);
  for (const table of document.querySelectorAll('table')) {
    if (table.querySelector('table')) continue;
    const rows = [...table.rows]; let hdr = -1, ki = -1, ci = -1;
    for (let r = 0; r < rows.length && hdr < 0; r++) {
      const cells = [...rows[r].cells].map(c => L.norm(c.innerText));
      ki = cells.findIndex(c => keyCols.includes(c)); ci = cells.findIndex(c => cols.includes(c));
      if (ki >= 0 && ci >= 0) hdr = r;
    }
    if (hdr < 0) continue;
    for (let r = hdr + 1; r < rows.length; r++) {
      const cells = [...rows[r].cells];
      if (cells.length > Math.max(ki, ci) && L.norm(cells[ki].innerText) === key) return L.clean(cells[ci].innerText);
    }
  }
  return null;
}"""

MASK_JS = "(args) => {" + CUA_LIB + r"""
  const want = args.labels.map(L.norm); let n = 0;
  const mark = e => { if (e && !e.hasAttribute('data-cua-mask')) { e.setAttribute('data-cua-mask', '1'); n++; } };
  document.querySelectorAll('input[type=password]').forEach(mark);
  for (const td of document.querySelectorAll('td, th')) {
    if (td.querySelector('table')) continue;
    if (want.includes(L.norm(td.innerText))) { let x = td.nextElementSibling; while (x && !L.clean(x.innerText)) x = x.nextElementSibling; mark(x); }
    if (/\b\d{3}-\d{2}-\d{4}\b/.test(td.innerText)) mark(td);
  }
  return n;
}"""

HUMAN_RECORDER_JS = "(() => { if (window.__cuaRec) return; window.__cuaRec = true;" + CUA_LIB + r"""
  const send = (type, el, extra) => { try {
    const r = L.role(el); if (!r || !window.__cuaHuman) return;
    window.__cuaHuman(Object.assign({type, role: r, name: L.name(el), nth: L.nthOf(el), frame: window.name || null,
                                     path: location.pathname, css: L.css(el)}, extra));
  } catch (e) {} };
  document.addEventListener('click', e => {
    const el = e.target.closest ? e.target.closest(L.SEL) : null; if (!el) return;
    const t = el.tagName, ty = (el.type || '').toLowerCase();
    if (t === 'SELECT' || t === 'TEXTAREA' || (t === 'INPUT' && !['submit', 'button', 'reset', 'image', 'checkbox', 'radio'].includes(ty))) return;
    send('click', el, {});
  }, true);
  document.addEventListener('change', e => {
    const el = e.target, t = el.tagName;
    if (t === 'SELECT') send('select', el, {value: el.selectedIndex >= 0 ? L.clean(el.options[el.selectedIndex].text) : null});
    else if (t === 'INPUT' || t === 'TEXTAREA') {
      const secret = (el.type || '').toLowerCase() === 'password';
      send('fill', el, {value: secret ? null : el.value, secret});
    }
  }, true);
})();"""


def render_pattern(pattern: str, params: dict[str, str]) -> str:
    return re.sub(r"\{([a-z_][a-z0-9_]*)\}", lambda m: params.get(m.group(1), m.group(0)), pattern)


def _names(name: str, aliases: dict[str, str]) -> list[str]:
    out = [name]
    if name in aliases:
        out.append(aliases[name])
    return out


class WebSurface:
    def __init__(self, base_url: str, policy: Policy, headless: bool = True, slow_mo: int = 0):
        self.base_url = base_url.rstrip("/")
        self.policy = policy
        self.violations: list[dict[str, str]] = []
        self._nav_count = 0
        self._seq = 0
        self._human_cb: Callable[[dict[str, Any]], None] | None = None
        self._pw = sync_playwright().start()
        self._browser = self._launch(headless, slow_mo)
        self._ctx = self._browser.new_context(viewport={"width": 1100, "height": 720})
        self._ctx.route("**/*", self._guard)
        self._ctx.expose_binding("__cuaHuman", self._on_human)
        self._ctx.add_init_script(HUMAN_RECORDER_JS)
        self.page: Page = self._ctx.new_page()
        self.page.on("framenavigated", lambda _f: self._bump())
        self._pending: set[Any] = set()
        self.page.on("request", lambda r: self._pending.add(r) if r.is_navigation_request() else None)
        self.page.on("requestfinished", lambda r: self._pending.discard(r))
        self.page.on("requestfailed", lambda r: self._pending.discard(r))

    def _launch(self, headless: bool, slow_mo: int):
        if not headless:
            # A visible window: prefer the locally installed Edge (some endpoint security blocks the bundled
            # Chromium from opening a window); fall back to the bundled browser.
            try:
                return self._pw.chromium.launch(channel="msedge", headless=False, slow_mo=slow_mo)
            except PWError:
                pass
        return self._pw.chromium.launch(headless=headless, slow_mo=slow_mo)

    # ------------------------------------------------------------------ guards / hooks
    def _bump(self) -> None:
        self._nav_count += 1

    def _guard(self, route, request) -> None:
        ok, why = self.policy.url_allowed(request.url)
        if not ok and (request.is_navigation_request() or urlsplit(request.url).netloc not in self.base_url):
            self.violations.append({"url": request.url, "reason": why})
            route.abort("blockedbyclient")
            return
        route.continue_()

    def _on_human(self, source, payload: dict[str, Any]) -> None:
        if self._human_cb:
            self._human_cb(payload)

    def on_human_action(self, cb: Callable[[dict[str, Any]], None] | None) -> None:
        self._human_cb = cb

    def navigation_count(self) -> int:
        return self._nav_count

    def navigation_pending(self) -> bool:
        return bool(self._pending)

    def pause(self, ms: int) -> None:
        # page.wait_for_timeout (unlike time.sleep) keeps pumping Playwright events, e.g. human-action bindings.
        self.page.wait_for_timeout(ms)

    # ------------------------------------------------------------------ frames
    def _frame(self, name: str | None) -> Frame | None:
        if name is None:
            return self.page.main_frame
        return next((f for f in self.page.frames if f.name == name), None)

    def _eval(self, frame: Frame, js: str, args: Any) -> Any:
        try:
            return frame.evaluate(js, args)
        except PWError:
            return None  # frame mid-navigation / detached; callers poll

    # ------------------------------------------------------------------ perception
    def open(self, route: str) -> None:
        self.page.goto(self.base_url + route, wait_until="load")

    def navigate(self, route: str) -> None:
        self.page.goto(self.base_url + route, wait_until="load")

    def observe(self, with_elements: bool = True) -> Observation:
        self._seq += 1
        frames, elements = [], []
        for i, f in enumerate(self.page.frames):
            fname = None if f == self.page.main_frame else (f.name or f"frame{i}")
            data = self._eval(f, SNAPSHOT_JS, {"withElements": with_elements, "prefix": f"e{i}_"})
            if data is None:
                frames.append(FrameState(fname, urlsplit(f.url).path, "", ""))
                continue
            frames.append(FrameState(fname, data["path"], data["title"], data["text"]))
            for e in data["elements"]:
                elements.append(UIElement(frame=fname, **e))
        return Observation(frames, elements)

    def resolve(self, target: Target, aliases: dict[str, str]) -> Resolved:
        frame = self._frame(target.frame)
        if frame is None:
            raise TargetError("target_not_found", f"frame '{target.frame}' not present")
        names = _names(target.name, aliases)
        prefix = f"r{self._seq}_{id(target) % 997}_"
        self._seq += 1
        hits = self._eval(frame, RESOLVE_JS, {"role": target.role, "names": names, "contains": False, "prefix": prefix}) or []
        if len(hits) > target.nth:
            return Resolved(self._el(target.frame, target.role, hits[target.nth]), "primary")
        for fb in target.fallbacks:
            if fb.strategy == "name_contains":
                hits = self._eval(frame, RESOLVE_JS, {"role": target.role, "names": [fb.value], "contains": True, "prefix": prefix + "c"}) or []
                if len(hits) == 1:
                    return Resolved(self._el(target.frame, target.role, hits[0]), "name_contains")
                if len(hits) > 1:
                    raise TargetError("target_ambiguous", f"{target.describe()}: fallback name_contains '{fb.value}' matched {len(hits)} controls")
            elif fb.strategy == "css":
                loc = frame.locator(fb.value)
                try:
                    n = loc.count()
                except PWError:
                    n = 0
                if n == 1:
                    ref = prefix + "css"
                    loc.evaluate("(el, r) => el.setAttribute('data-cua-ref', r)", ref)
                    return Resolved(UIElement(ref, target.frame, target.role, target.name, 0, css=fb.value), "css")
                if n > 1:
                    raise TargetError("target_ambiguous", f"{target.describe()}: fallback css '{fb.value}' matched {n} controls")
        raise TargetError("target_not_found", f"no control matches {target.describe()}")

    @staticmethod
    def _el(frame: str | None, role: str, h: dict[str, Any]) -> UIElement:
        return UIElement(h["ref"], frame, role, h["name"], h["nth"], tag=h["tag"], css=h["css"])

    def _loc(self, r: Resolved):
        frame = self._frame(r.element.frame)
        if frame is None:
            raise TargetError("target_not_found", f"frame '{r.element.frame}' disappeared")
        return frame.locator(f'[data-cua-ref="{r.element.ref}"]')

    # ------------------------------------------------------------------ actions
    def click(self, r: Resolved) -> None:
        self._loc(r).click(timeout=5000, no_wait_after=True)

    def fill(self, r: Resolved, value: str) -> None:
        self._loc(r).fill(value, timeout=5000)

    def select(self, r: Resolved, label: str) -> None:
        self._loc(r).select_option(label=label, timeout=5000)

    def press(self, r: Resolved, key: str) -> None:
        self._loc(r).press(key, timeout=5000, no_wait_after=True)

    # ------------------------------------------------------------------ reading
    def extract(self, source, row_key: str | None, aliases: dict[str, str]) -> str | None:
        frame = self._frame(source.frame)
        if frame is None:
            return None
        if isinstance(source, LabeledValue):
            return self._eval(frame, LABELED_VALUE_JS, {"labels": _names(source.label, aliases)})
        if isinstance(source, TableCell):
            return self._eval(frame, TABLE_CELL_JS, {
                "keyCols": _names(source.row_key_column, aliases), "cols": _names(source.column, aliases), "key": row_key or ""})
        if isinstance(source, TextRegex):
            text = self._eval(frame, "() => document.body ? document.body.innerText : ''", None) or ""
            m = re.search(source.pattern, text)
            return (m.group(1) if m.groups() else m.group(0)).strip() if m else None
        raise ValueError(f"unknown extraction {source}")

    def check(self, cond: Condition, params: dict[str, str], aliases: dict[str, str], obs: Observation | None = None) -> bool:
        obs = obs or self.observe(with_elements=False)
        if isinstance(cond, RouteCond):
            f = obs.frame(cond.frame)
            return bool(f) and fnmatch.fnmatchcase(f.path, render_pattern(cond.pattern, params))
        if isinstance(cond, TitleCond):
            f = obs.frame(cond.frame)
            return bool(f) and f.title.strip().lower() == render_pattern(cond.equals, params).strip().lower()
        if isinstance(cond, TextCond):
            frames = [obs.frame(cond.frame)] if cond.frame else obs.frames
            for f in filter(None, frames):
                if cond.contains and render_pattern(cond.contains, params).lower() in f.text.lower():
                    return True
                if cond.regex and re.search(cond.regex, f.text):
                    return True
            return False
        if isinstance(cond, ElementCond):
            try:
                self.resolve(cond.target, aliases)
                return True
            except TargetError:
                return False
        raise ValueError(cond)

    # ------------------------------------------------------------------ evidence
    def screenshot(self, path: str, pii_labels: list[str]) -> None:
        masks = []
        for f in self.page.frames:
            if self._eval(f, MASK_JS, {"labels": pii_labels}):
                masks.append(f.locator("[data-cua-mask]"))
        try:
            self.page.screenshot(path=path, mask=masks, mask_color="#222")
        except PWError:
            pass

    def dom_digest(self, obs: Observation) -> str:
        """Text + control inventory per frame (callers redact before writing)."""
        out = []
        for f in obs.frames:
            out.append(f"=== frame {f.name or '(top)'}  path={f.path}  title={f.title!r}\n{f.text}\n")
        out.append("=== controls")
        out += [f"{e.frame}: {e.role} {e.name!r} nth={e.nth} css={e.css}" for e in obs.elements]
        return "\n".join(out)

    def close(self) -> None:
        try:
            self._ctx.close()
            self._browser.close()
        finally:
            self._pw.stop()


def elements_json(obs: Observation) -> str:
    return json.dumps([e.__dict__ for e in obs.elements])
