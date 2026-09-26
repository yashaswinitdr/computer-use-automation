"""Turn a successful discovery trace into a Capability artifact -- decoupled from the model transcript.

What is kept: the actions that succeeded (model- or human-performed), how each control was identified, the
state each action led to (as a postcondition), declared outputs and how to read them, and a final success
checkpoint. What is dropped: rationales beyond a one-line intent, raw observations, rejected/blocked proposals.

Parameterisation: a typed value becomes an input reference if the model tagged it with ``param`` or it equals a
parameter value exactly. Routes, titles and success text are canonicalised the same way (/mbr/10042 ->
/mbr/{member_id}), so the artifact never contains the example values used during discovery.
"""

from __future__ import annotations

import re
from typing import Any

from cua.agent.discovery import DiscoveryOutcome, TraceEntry
from cua.schema import (
    AppProfile, AppRef, Capability, Checkpoint, Click, Condition, Extract, Fallback, Fill, LabeledValue,
    LiteralValue, OutcomeSpec, ParamSpec, ParamValue, Press, Provenance, RouteCond, Select, Step, TableCell,
    Target, TextCond, TextRegex, TitleCond,
)


def canonicalize(text: str, params: dict[str, str]) -> str:
    for name, value in sorted(params.items(), key=lambda kv: -len(kv[1])):
        if value and len(value) >= 3:
            text = text.replace(value, "{" + name + "}")
    return text


def infer_param_spec(name: str, example: str) -> ParamSpec:
    if re.fullmatch(r"\d+", example):
        return ParamSpec(type="string", pattern=r"^\d{1,12}$", description=f"{name.replace('_', ' ')} (digits)")
    if re.fullmatch(r"-?\d+\.\d+", example):
        return ParamSpec(type="decimal", description=name.replace("_", " "), sensitivity="internal")
    return ParamSpec(type="string", description=name.replace("_", " "))


def _input_specs(trace: list[TraceEntry], params: dict[str, str], used: set[str]) -> dict[str, ParamSpec]:
    specs = {n: infer_param_spec(n, params[n]) for n in params if n in used}
    for e in trace:  # a parameter fed into a dropdown is an enum of the options that were on screen
        if e.action == "select" and e.options:
            name = e.param or next((k for k, v in params.items() if v == e.text), None)
            if name in specs:
                specs[name] = ParamSpec(type="enum", enum=e.options, sensitivity="internal",
                                        description=f"{name.replace('_', ' ')} (one of the host's options)")
    return specs


def _value(e: TraceEntry, params: dict[str, str]):
    if e.param and e.param in params:
        return ParamValue(name=e.param)
    for name, v in params.items():
        if e.text == v:
            return ParamValue(name=name)
    return LiteralValue(value=e.text or "")


def _target(e: TraceEntry, params: dict[str, str]) -> Target:
    fallbacks = []
    # Only *semantic* CSS (field name / button caption) is kept as a fallback. Positional paths
    # (nth-of-type chains) are dropped on purpose: when layout shifts they silently hit the wrong control.
    if e.css and not e.css.startswith("body >") and not any(v in e.css for v in params.values() if len(v) >= 3):
        fallbacks.append(Fallback(strategy="css", value=e.css))
    return Target(frame=e.frame, role=e.role or "button", name=e.name or "", nth=e.nth, fallbacks=fallbacks)


def _expect(e: TraceEntry, params: dict[str, str]) -> Checkpoint | None:
    if not e.after:
        return None
    before = {n: (p, t) for n, p, t in e.before}
    conds: list[Condition] = []
    for name, path, title in e.after:
        if before.get(name) == (path, title) and e.before:
            continue
        if name == "nav":
            continue
        conds.append(RouteCond(frame=name, pattern=canonicalize(path, params)))
        if title:
            conds.append(TitleCond(frame=name, equals=canonicalize(title, params)))
    return Checkpoint(all_of=conds, timeout_ms=10_000) if conds else None


def build_capability(outcome: DiscoveryOutcome, *, capability_id: str, title: str, description: str,
                     goal_template: str, params: dict[str, str], profile: AppProfile, run_id: str, model: str,
                     version: str = "1.0.0") -> Capability:
    steps: list[Step] = []
    for i, e in enumerate(outcome.trace, start=1):
        sid = f"s{i}"
        src = "human" if e.source == "human" else "model"
        if e.action == "extract":
            x: dict[str, Any] = dict(e.extract["source"])
            kind = x.pop("kind")
            if kind == "table_cell":
                rk = x["row_key"]["value"]
                pn = e.extract.get("row_key_param")
                x["row_key"] = ParamValue(name=pn) if pn in params else (
                    ParamValue(name=next(k for k, v in params.items() if v == rk)) if rk in params.values() else LiteralValue(value=rk))
                source = TableCell(**x)
            elif kind == "labeled_value":
                source = LabeledValue(**x)
            else:
                source = TextRegex(**x)
            out = e.extract["output"]
            steps.append(Step(id=sid, intent=f"Read output '{out}'", action=Extract(output=out, source=source), source=src))
            continue
        tgt = _target(e, params)
        if e.action == "click":
            action = Click(target=tgt)
            intent = f'Activate {tgt.role} "{tgt.name}"'
        elif e.action == "fill":
            action = Fill(target=tgt, value=_value(e, params))
            shown = "{" + action.value.name + "}" if isinstance(action.value, ParamValue) else "a constant"
            intent = f'Enter {shown} into "{tgt.name}"'
        elif e.action == "select":
            action = Select(target=tgt, value=_value(e, params))
            intent = f'Choose option in "{tgt.name}"'
        else:
            action = Press(target=tgt, key=e.key or "Enter")
            intent = f'Press {e.key} in "{tgt.name}"'
        steps.append(Step(id=sid, intent=intent, action=action, risk=e.risk, expect=_expect(e, params), source=src))

    success: list[Condition] = []
    for name, path, title_ in outcome.final_state:
        if name == "nav":
            continue
        success.append(RouteCond(frame=name, pattern=canonicalize(path, params)))
        if title_:
            success.append(TitleCond(frame=name, equals=canonicalize(title_, params)))
    if outcome.success_text:
        success.append(TextCond(frame=None, contains=canonicalize(outcome.success_text, params)))

    used = {s.action.value.name for s in steps if isinstance(getattr(s.action, "value", None), ParamValue)}
    used |= {s.action.source.row_key.name for s in steps if isinstance(s.action, Extract)
             and isinstance(getattr(s.action.source, "row_key", None), ParamValue)}
    blob = " ".join(c.model_dump_json() for c in success) + " ".join(s.model_dump_json() for s in steps)
    used |= {n for n in params if "{" + n + "}" in blob}

    return Capability(
        id=capability_id,
        version=version,
        status="draft",
        title=title,
        description=description,
        app=AppRef(profile=profile.id, profile_version=profile.version),
        inputs=_input_specs(outcome.trace, params, used),
        outputs=outcome.output_specs,
        outcomes=[OutcomeSpec(code=k.code, description=k.description) for k in profile.known_states if k.classification == "business"],
        steps=steps,
        success=Checkpoint(all_of=success, timeout_ms=10_000),
        provenance=Provenance(method="llm_discovery", discovery_run_id=run_id, model=model, goal_template=goal_template,
                              human_steps=sum(1 for s in steps if s.source == "human")),
    )
