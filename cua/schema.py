"""Typed, versioned data model for capabilities (recorded flows), app profiles, tenants and replay results.

Three layers, deliberately separated (see REPORT.md section 2):

* ``Capability``   -- one reusable, agent-invocable flow (what an AI agent calls). Portable across tenants.
* ``AppProfile``   -- knowledge about a vendor *product/version*: session flows (sign-on), and the taxonomy of
                      runtime states (business outcomes / recoverable / hard failures) with their detectors.
                      Shared by every capability and every tenant running that product.
* ``TenantConfig`` -- where one institution's instance lives, which secrets it uses, and small overlays
                      (caption aliases) for how that tenant configured the product.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

CAPABILITY_SCHEMA = "cua.capability/v1"
PROFILE_SCHEMA = "cua.app_profile/v1"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------------------------ values
class Sensitivity(str, Enum):  # noqa: UP042 (str mixin keeps JSON values plain strings)
    public = "public"  # safe to log verbatim
    internal = "internal"  # logged verbatim inside the platform, never sent to third parties
    pii = "pii"  # masked in logs/evidence (e.g. member numbers, names)
    secret = "secret"  # never logged, never stored in artifacts; only ever a reference


class LiteralValue(Strict):
    kind: Literal["literal"] = "literal"
    value: str


class ParamValue(Strict):
    kind: Literal["param"] = "param"
    name: str


class SecretValue(Strict):
    """Resolved at execution time from the tenant's secret provider. The artifact only holds the name."""

    kind: Literal["secret"] = "secret"
    name: str


ValueExpr = Annotated[Union[LiteralValue, ParamValue, SecretValue], Field(discriminator="kind")]


class ParamSpec(Strict):
    type: Literal["string", "integer", "decimal", "boolean", "enum"]
    description: str = ""
    required: bool = True
    pattern: str | None = None
    enum: list[str] | None = None
    minimum: float | None = None
    maximum: float | None = None
    sensitivity: Sensitivity = Sensitivity.pii

    def validate_value(self, raw: Any) -> str:
        """Return the canonical string form or raise ValueError with a caller-facing message."""
        s = str(raw).strip()
        if self.type == "integer" and not re.fullmatch(r"-?\d+", s):
            raise ValueError("must be an integer")
        if self.type == "decimal":
            try:
                float(s)
            except ValueError:
                raise ValueError("must be a decimal number") from None
        if self.type == "boolean" and s.lower() not in ("true", "false"):
            raise ValueError("must be true or false")
        if self.type == "enum" and self.enum and s not in self.enum:
            raise ValueError(f"must be one of {self.enum}")
        if self.pattern and not re.fullmatch(self.pattern, s):
            raise ValueError(f"must match {self.pattern}")
        if self.type in ("integer", "decimal"):
            v = float(s)
            if self.minimum is not None and v < self.minimum:
                raise ValueError(f"must be >= {self.minimum}")
            if self.maximum is not None and v > self.maximum:
                raise ValueError(f"must be <= {self.maximum}")
        return s


class OutputSpec(Strict):
    type: Literal["string", "integer", "money", "date"]
    description: str = ""
    sensitivity: Sensitivity = Sensitivity.internal


# ------------------------------------------------------------------------------------------------ targeting
class Fallback(Strict):
    """Secondary ways to find the same control, tried in order only if the primary (role+name) fails.

    Any fallback hit is reported as *drift* in the replay result so a human can re-baseline the artifact.
    """

    strategy: Literal["name_contains", "css"]
    value: str


class Target(Strict):
    """How a control is identified: by what a human operator perceives, not by DOM structure.

    Primary key = (frame, role, accessible name, nth). The accessible name is computed by the perception layer
    the same way for discovery and replay (aria-label > <label> > button caption/link text > caption in the
    neighbouring table cell > placeholder/title). On a desktop surface the same triple maps to
    (window, UIA ControlType, UIA Name).
    """

    frame: str | None = Field(None, description="Named frame/window the control lives in; null = top document")
    role: str
    name: str
    nth: int = Field(0, description="Index among controls with the same role+name in the frame (document order)")
    fallbacks: list[Fallback] = []

    def describe(self) -> str:
        f = f"{self.frame}:" if self.frame else ""
        n = f"[{self.nth}]" if self.nth else ""
        return f'{f}{self.role} "{self.name}"{n}'


# ------------------------------------------------------------------------------------------------ conditions
class RouteCond(Strict):
    kind: Literal["route"] = "route"
    frame: str | None = None
    pattern: str = Field(description="Path pattern; {param} placeholders are filled from inputs, * is a wildcard")


class TitleCond(Strict):
    kind: Literal["title"] = "title"
    frame: str | None = None
    equals: str


class TextCond(Strict):
    kind: Literal["text"] = "text"
    frame: str | None = Field(None, description="null = any frame")
    contains: str | None = None
    regex: str | None = None

    @model_validator(mode="after")
    def _one(self):
        if bool(self.contains) == bool(self.regex):
            raise ValueError("text condition needs exactly one of contains/regex")
        return self


class ElementCond(Strict):
    kind: Literal["element"] = "element"
    target: Target


Condition = Annotated[Union[RouteCond, TitleCond, TextCond, ElementCond], Field(discriminator="kind")]


class Checkpoint(Strict):
    all_of: list[Condition]
    timeout_ms: int = 10_000


# ------------------------------------------------------------------------------------------------ extraction
class LabeledValue(Strict):
    """Value in the cell next to a caption cell ("Name:" | "DOE, JANE")."""

    kind: Literal["labeled_value"] = "labeled_value"
    frame: str | None = None
    label: str


class TableCell(Strict):
    """Cell at (row whose <row_key_column> equals <row_key>, column <column>). Survives row re-ordering."""

    kind: Literal["table_cell"] = "table_cell"
    frame: str | None = None
    row_key_column: str
    row_key: ValueExpr
    column: str


class TextRegex(Strict):
    kind: Literal["text_regex"] = "text_regex"
    frame: str | None = None
    pattern: str = Field(description="First capture group is the value")


ExtractionSource = Annotated[Union[LabeledValue, TableCell, TextRegex], Field(discriminator="kind")]


# ------------------------------------------------------------------------------------------------ actions
class Click(Strict):
    type: Literal["click"] = "click"
    target: Target


class Fill(Strict):
    type: Literal["fill"] = "fill"
    target: Target
    value: ValueExpr


class Select(Strict):
    type: Literal["select"] = "select"
    target: Target
    value: ValueExpr = Field(description="Matched against the visible option label")


class Press(Strict):
    type: Literal["press"] = "press"
    target: Target
    key: str


class Navigate(Strict):
    type: Literal["navigate"] = "navigate"
    route: str = Field(description="Relative to the tenant base_url; may contain {param}")


class Extract(Strict):
    type: Literal["extract"] = "extract"
    output: str
    source: ExtractionSource


Action = Annotated[Union[Click, Fill, Select, Press, Navigate, Extract], Field(discriminator="type")]

Risk = Literal["read", "write", "irreversible"]
RISK_ORDER = {"read": 0, "write": 1, "irreversible": 2}


class Step(Strict):
    id: str
    intent: str = Field(description="Human-readable purpose of the step (for reviewers and error messages)")
    action: Action
    risk: Risk = "read"
    expect: Checkpoint | None = Field(None, description="Postcondition asserted after the action")
    source: Literal["model", "human", "authored"] = "model"


class OutcomeSpec(Strict):
    code: str
    description: str


class AppRef(Strict):
    profile: str
    profile_version: str


class Provenance(Strict):
    method: Literal["llm_discovery", "authored"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    discovery_run_id: str | None = None
    model: str | None = None
    goal_template: str | None = None
    human_steps: int = 0
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None


class Capability(Strict):
    schema_: Literal["cua.capability/v1"] = Field(CAPABILITY_SCHEMA, alias="schema")
    id: str = Field(pattern=r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    status: Literal["draft", "approved", "deprecated"] = "draft"
    title: str
    description: str
    app: AppRef
    surface: Literal["web"] = "web"
    entry_route: str = "/"
    requires_session: bool = True
    inputs: dict[str, ParamSpec] = {}
    outputs: dict[str, OutputSpec] = {}
    outcomes: list[OutcomeSpec] = Field(default=[], description="Business outcomes the caller may receive")
    steps: list[Step]
    success: Checkpoint
    provenance: Provenance

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @property
    def risk(self) -> Risk:
        return max((s.risk for s in self.steps), key=RISK_ORDER.__getitem__, default="read")

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @model_validator(mode="after")
    def _consistent(self):
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique")
        extracted = [s.action.output for s in self.steps if isinstance(s.action, Extract)]
        if sorted(extracted) != sorted(self.outputs):
            raise ValueError(f"every output needs exactly one extract step (outputs={list(self.outputs)}, extracted={extracted})")
        for name in _param_refs(self):
            if name not in self.inputs:
                raise ValueError(f"step references undeclared input '{name}'")
        return self

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, indent=2, exclude_none=True)


def _param_refs(obj: Any) -> set[str]:
    """All {param} placeholders and ParamValue references anywhere inside a model."""
    found: set[str] = set()
    if isinstance(obj, ParamValue):
        found.add(obj.name)
    elif isinstance(obj, BaseModel):
        for k in type(obj).model_fields:
            v = getattr(obj, k)
            if isinstance(v, str) and k in ("pattern", "route", "contains"):
                found |= set(re.findall(r"\{([a-z_][a-z0-9_]*)\}", v))
            else:
                found |= _param_refs(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found |= _param_refs(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _param_refs(v)
    return found


# ------------------------------------------------------------------------------------------------ app profile
class Recovery(Strict):
    kind: Literal["click", "wait", "run_flow"]
    target: Target | None = None
    wait_ms: int = 1000
    flow: str | None = None
    then: Literal["resume", "restart"] = "resume"
    max_attempts: int = 2


class KnownState(Strict):
    """A runtime condition the replay engine recognises, and what it means for the caller."""

    code: str
    classification: Literal["business", "recoverable", "hard_failure"]
    description: str
    detect_any: list[Condition]
    message_regex: str | None = Field(None, description="Captures the app's own message for the caller")
    recovery: Recovery | None = None


class Flow(Strict):
    steps: list[Step]
    success: Checkpoint


class AppProfile(Strict):
    schema_: Literal["cua.app_profile/v1"] = Field(PROFILE_SCHEMA, alias="schema")
    id: str
    version: str
    vendor_product: str
    description: str = ""
    signed_out_detect: list[Condition] = []
    sign_on_flow: str | None = None
    flows: dict[str, Flow] = {}
    known_states: list[KnownState] = []
    pii_labels: list[str] = Field(default=[], description="Captions whose neighbouring values are masked in screenshots")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ------------------------------------------------------------------------------------------------ tenant
class TenantConfig(Strict):
    tenant_id: str
    app_profile: str
    base_url: str
    secrets: dict[str, str] = Field(default={}, description="secret name -> environment variable holding it")
    aliases: dict[str, str] = Field(
        default={}, description="Overlay: canonical caption (as recorded) -> this tenant's caption"
    )


# ------------------------------------------------------------------------------------------------ results
class StateView(Strict):
    """Redacted view of what the surface looked like (for errors and intervention requests)."""

    frames: list[dict[str, Any]] = []


class Failure(Strict):
    code: Literal[
        "invalid_request", "policy_violation", "target_not_found", "target_ambiguous", "checkpoint_timeout",
        "unrecognized_state", "host_error", "output_invalid", "approval_denied", "escalation_aborted",
        "recovery_exhausted", "surface_error",
    ]
    message: str
    step_id: str | None = None
    step_intent: str | None = None
    expected: str | None = None
    observed: StateView | None = None
    evidence: dict[str, str] = {}


class BusinessOutcome(Strict):
    code: str
    message: str | None = None
    step_id: str | None = None


class RecoveryRecord(Strict):
    code: str
    step_id: str | None
    action: str


class DriftRecord(Strict):
    step_id: str
    target: str
    strategy: str


class ReplayResult(Strict):
    """The contract returned to the calling agent. Exactly one of outputs / outcome / failure is meaningful."""

    status: Literal["succeeded", "business_outcome", "failed"]
    capability: str
    run_id: str
    tenant_id: str
    outputs: dict[str, Any] = {}
    outcome: BusinessOutcome | None = None
    failure: Failure | None = None
    recoveries: list[RecoveryRecord] = []
    drift: list[DriftRecord] = []
    interventions: list[str] = []
    steps_completed: int = 0
    duration_ms: int = 0
