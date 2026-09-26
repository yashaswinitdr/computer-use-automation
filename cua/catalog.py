"""Loading artifacts/profiles/tenants, and exposing approved capabilities as agent-callable tools (stretch goal)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cua.schema import AppProfile, Capability, TenantConfig

ROOT = Path(__file__).resolve().parent.parent


def load_capability(path_or_id: str, caps_dir: Path = ROOT / "capabilities", approved_only: bool = False) -> Capability:
    p = Path(path_or_id)
    if p.suffix == ".json" and p.exists():
        return Capability.model_validate_json(p.read_text(encoding="utf-8"))
    cid, _, ver = path_or_id.partition("@")
    found = [c for c in all_capabilities(caps_dir) if c.id == cid and (not ver or c.version == ver)
             and (c.status == "approved" or not approved_only)]
    if not found:
        raise FileNotFoundError(f"no {'approved ' if approved_only else ''}capability {path_or_id} in {caps_dir}")
    return max(found, key=lambda c: tuple(int(x) for x in c.version.split(".")))


def all_capabilities(caps_dir: Path = ROOT / "capabilities") -> list[Capability]:
    return [Capability.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(caps_dir.glob("*.json"))]


def capability_path(cap: Capability, caps_dir: Path = ROOT / "capabilities") -> Path:
    return caps_dir / f"{cap.id}@{cap.version}.json"


def load_profile(ref: str, profiles_dir: Path = ROOT / "profiles") -> AppProfile:
    pid, _, ver = ref.partition("@")
    return AppProfile.model_validate_json((profiles_dir / f"{pid}-{ver}.json").read_text(encoding="utf-8"))


def load_tenant(path: str) -> TenantConfig:
    return TenantConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _json_type(t: str) -> dict[str, Any]:
    return {"integer": {"type": "integer"}, "decimal": {"type": "string", "pattern": r"^-?\d+(\.\d+)?$"},
            "boolean": {"type": "boolean"}}.get(t, {"type": "string"})


def tool_definition(cap: Capability) -> dict[str, Any]:
    props, required = {}, []
    for n, spec in cap.inputs.items():
        s = _json_type(spec.type) | {"description": spec.description}
        if spec.pattern:
            s["pattern"] = spec.pattern
        if spec.enum:
            s["enum"] = spec.enum
        props[n] = s
        if spec.required:
            required.append(n)
    outs = ", ".join(f"{n} ({o.type})" for n, o in cap.outputs.items()) or "none"
    outcomes = ", ".join(o.code for o in cap.outcomes) or "none"
    return {
        "name": cap.id.replace(".", "__"),
        "description": (f"{cap.title}. {cap.description} Returns outputs: {outs}. Possible business outcomes "
                        f"(not errors): {outcomes}. Risk: {cap.risk}. Version {cap.version}."),
        "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False},
        "x-capability": cap.ref,
        "x-returns": {n: o.model_dump() for n, o in cap.outputs.items()},
    }


def catalog(include_drafts: bool = False, caps_dir: Path = ROOT / "capabilities") -> list[dict[str, Any]]:
    latest: dict[str, Capability] = {}
    for c in all_capabilities(caps_dir):
        if c.status == "deprecated" or (c.status == "draft" and not include_drafts):
            continue
        cur = latest.get(c.id)
        if cur is None or tuple(map(int, c.version.split("."))) > tuple(map(int, cur.version.split("."))):
            latest[c.id] = c
    return [tool_definition(c) for c in latest.values()]


def dumps(o: Any) -> str:
    return json.dumps(o, indent=2, default=str)
