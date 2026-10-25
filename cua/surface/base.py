"""The seam between "how we perceive / act on a surface" and "the recorded flow".

Everything above this interface (agent loop, recorder, replay engine, artifacts) speaks only in terms of
frames/windows, roles, accessible names, conditions and extraction sources. A surface implementation turns those
into concrete operations. ``WebSurface`` (Playwright, DOM + computed accessible names) is the one implemented here;
a ``UIASurface`` for Windows desktop apps or a ``VisionSurface`` for pixel-only (Citrix/terminal) screens would
implement the same protocol -- see REPORT.md section 4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from cua.schema import Condition, ExtractionSource, Target


@dataclass
class UIElement:
    ref: str  # ephemeral handle valid for one observation (e.g. "e12")
    frame: str | None
    role: str
    name: str
    nth: int
    tag: str = ""
    value: str | None = None
    options: list[str] = field(default_factory=list)
    secret: bool = False  # password-type control: value never read, never shown to the model
    css: str | None = None  # structural fallback


@dataclass
class FrameState:
    name: str | None
    path: str
    title: str
    text: str


@dataclass
class Observation:
    frames: list[FrameState]
    elements: list[UIElement] = field(default_factory=list)

    def frame(self, name: str | None) -> FrameState | None:
        return next((f for f in self.frames if f.name == name), None)

    def signature(self) -> str:
        return "|".join(f"{f.name}:{f.path}:{f.title}" for f in self.frames)


@dataclass
class Resolved:
    element: UIElement
    strategy: str  # "primary" or the fallback strategy that matched


class TargetError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code  # target_not_found | target_ambiguous


class Surface(Protocol):
    def open(self, route: str) -> None: ...
    def observe(self, with_elements: bool = True) -> Observation: ...
    def resolve(self, target: Target, aliases: dict[str, str]) -> Resolved: ...
    def click(self, r: Resolved) -> None: ...
    def fill(self, r: Resolved, value: str) -> None: ...
    def select(self, r: Resolved, label: str) -> None: ...
    def press(self, r: Resolved, key: str) -> None: ...
    def navigate(self, route: str) -> None: ...
    def extract(self, source: ExtractionSource, row_key: str | None, aliases: dict[str, str]) -> str | None: ...
    def check(self, cond: Condition, params: dict[str, str], aliases: dict[str, str]) -> bool: ...
    def navigation_count(self) -> int: ...
    def navigation_pending(self) -> bool: ...
    def pause(self, ms: int) -> None: ...
    def screenshot(self, path: str, pii_labels: list[str]) -> None: ...
    def on_human_action(self, cb: Callable[[dict[str, Any]], None]) -> None: ...
    def close(self) -> None: ...
