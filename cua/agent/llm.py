"""Provider-neutral "decide the next action" interface.

The agent loop is stateless per turn: each call gets the goal, a compact action history and the *current*
observation, and must answer with exactly one tool call. That keeps prompts small and bounded, makes every
decision independently loggable/replayable for debugging, and makes providers interchangeable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    text: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    pass


class LLM(Protocol):
    model: str

    def decide(self, system: str, user: str, tools: list[dict[str, Any]], obs: Any = None) -> ToolCall: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or "claude-opus-5"

    def decide(self, system: str, user: str, tools: list[dict[str, Any]], obs: Any = None) -> ToolCall:
        a = self._anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools],
                tool_choice={"type": "auto"},
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.RateLimitError as e:
            raise LLMError(f"rate limited: {e}") from e
        except a.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request (refusal)")
        text = " ".join(b.text for b in resp.content if b.type == "text") or None
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        for b in resp.content:
            if b.type == "tool_use":
                return ToolCall(b.name, dict(b.input), text, usage)
        return ToolCall("none", {}, text, usage)


class OpenAILLM:
    def __init__(self, model: str | None = None):
        import openai

        self._openai = openai
        self.client = openai.OpenAI()
        self.model = model or "gpt-4.1"

    def decide(self, system: str, user: str, tools: list[dict[str, Any]], obs: Any = None) -> ToolCall:
        o = self._openai
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                tools=[{"type": "function", "function": t} for t in tools],
                tool_choice="required",
                parallel_tool_calls=False,
            )
        except o.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except o.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        msg = resp.choices[0].message
        usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens} if resp.usage else {}
        if not msg.tool_calls:
            return ToolCall("none", {}, msg.content, usage)
        tc = msg.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            return ToolCall("none", {}, f"unparseable tool arguments: {tc.function.arguments[:200]}", usage)
        return ToolCall(tc.function.name, args, msg.content, usage)


class ScriptedLLM:
    """Deterministic stand-in for offline tests/demos. NOT used for the evidence discovery run.

    Script entries may name a control by {"target": {"frame", "role", "name"}} instead of an ephemeral ref;
    it is mapped to the ref in the current observation, exactly as a model would pick one.
    """

    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)
        self.model = "scripted"

    @classmethod
    def from_file(cls, path: str) -> "ScriptedLLM":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def decide(self, system: str, user: str, tools: list[dict[str, Any]], obs: Any = None) -> ToolCall:
        if not self.script:
            return ToolCall("request_help", {"reason": "script exhausted"})
        entry = dict(self.script.pop(0))
        args = dict(entry.get("args", {}))
        tgt = args.pop("target", None)
        if tgt and obs is not None:
            for e in obs.elements:
                if e.role == tgt["role"] and e.name.lower() == tgt["name"].lower() and e.frame == tgt.get("frame", e.frame):
                    args["ref"] = e.ref
                    break
            else:
                args["ref"] = "missing"
        args.setdefault("rationale", "scripted")
        return ToolCall(entry["tool"], args)


def make_llm(provider: str | None, model: str | None, script: str | None = None) -> LLM:
    provider = provider or os.environ.get("CUA_PROVIDER") or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai")
    model = model or os.environ.get("CUA_MODEL")
    if provider == "scripted":
        if not script:
            raise LLMError("--script is required with provider=scripted")
        return ScriptedLLM.from_file(script)
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAILLM(model)
    raise LLMError(f"unknown provider {provider}")
