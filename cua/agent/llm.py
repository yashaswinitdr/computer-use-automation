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


def _parse_text_tool_call(text: str, names: set[str]) -> tuple[str, dict[str, Any]] | None:
    """Some local models write the tool call into the message text instead of the tool_calls field, as
    `finish {...}`, `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` or a bare JSON object.
    Accept it only if it names a known tool and its arguments are a JSON object; the decision is unchanged."""
    import re

    t = re.sub(r"</?tool_call>|```(?:json)?", "", text).strip()
    m = re.match(r"^([a-z_]+)\s*(\{.*\})\s*$", t, re.S)
    try:
        if m and m.group(1) in names:
            args = json.loads(m.group(2))
            return (m.group(1), args) if isinstance(args, dict) else None
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("name") in names:
        args = obj.get("arguments") or obj.get("parameters") or {}
        return (obj["name"], args) if isinstance(args, dict) else None
    return None


class OllamaLLM:
    """Local model served by Ollama (no API key; screen data never leaves the machine).

    Uses Ollama's native /api/chat so the context window can be raised: observations are a few thousand
    tokens and Ollama's default context would silently truncate them.
    """

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or "qwen2.5:7b"
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")

    def decide(self, system: str, user: str, tools: list[dict[str, Any]], obs: Any = None) -> ToolCall:
        import urllib.error
        import urllib.request

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "tools": [{"type": "function", "function": t} for t in tools],
            "stream": False,
            "options": {"num_ctx": 16384, "temperature": 0},
        }
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise LLMError(f"ollama error {e.code}: {e.read()[:200]!r}") from e
        except OSError as e:
            raise LLMError(f"ollama unreachable at {self.host}: {e}") from e
        msg = resp.get("message", {})
        usage = {"input_tokens": resp.get("prompt_eval_count"), "output_tokens": resp.get("eval_count")}
        calls = msg.get("tool_calls") or []
        if not calls:
            parsed = _parse_text_tool_call(msg.get("content") or "", {t["name"] for t in tools})
            if parsed:
                return ToolCall(parsed[0], parsed[1], msg.get("content"), usage | {"parsed_from_text": True})
            return ToolCall("none", {}, msg.get("content"), usage)
        fn = calls[0]["function"]
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return ToolCall("none", {}, f"unparseable tool arguments: {args[:200]}", usage)
        return ToolCall(fn["name"], args, msg.get("content"), usage)


class ScriptedLLM:
    """Deterministic stand-in for offline tests/demos. NOT used for the evidence discovery run.

    Script entries may name a control by {"target": {"frame", "role", "name"}} instead of an ephemeral ref;
    it is mapped to the ref in the current observation, exactly as a model would pick one.
    """

    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)
        self.model = "scripted"

    @classmethod
    def from_file(cls, path: str) -> ScriptedLLM:
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
    # Default: a local model through Ollama -- no API key, and screen data never leaves the machine.
    # Hosted providers are opt-in via --provider / CUA_PROVIDER.
    provider = provider or os.environ.get("CUA_PROVIDER") or "ollama"
    model = model or os.environ.get("CUA_MODEL")
    if provider == "scripted":
        if not script:
            raise LLMError("--script is required with provider=scripted")
        return ScriptedLLM.from_file(script)
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAILLM(model)
    if provider == "ollama":
        return OllamaLLM(model)
    raise LLMError(f"unknown provider {provider}")
