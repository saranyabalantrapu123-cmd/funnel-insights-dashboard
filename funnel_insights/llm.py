"""LLM access via LiteLLM, with a deterministic offline MockLLM.

* Model: env `LLM_MODEL` (default `anthropic/claude-sonnet-4-5`).
* Key: `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` for `openai/...` models).
* Offline demo mode: `LLM_MODEL=mock` or no key -> `MockLLM`. Callers pass a
  `mock_response` callable with every request; the MockLLM runs it to produce
  rule-based output in exactly the same message shape a real model would return,
  so the calling code path (including tool-calling loops) is identical offline.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"
OFFLINE_BANNER = "Running in offline demo mode (no LLM key) — outputs are rule-based"

Message = dict[str, Any]
MockFn = Callable[[list[Message]], Message]


@dataclass
class UsageTracker:
    """Token usage and estimated cost accumulated across calls in one run."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.calls} LLM call(s) · {self.prompt_tokens:,} prompt + "
                f"{self.completion_tokens:,} completion tokens · est. ${self.cost_usd:.4f}")


class LLM:
    """Common interface. `complete` returns {"content": str | None, "tool_calls": [...]}."""

    offline: bool = False
    model: str = ""

    def __init__(self) -> None:
        self.usage = UsageTracker()

    def complete(self, messages: list[Message], *, tools: list[dict] | None = None,
                 json_mode: bool = False, mock_response: MockFn | None = None) -> Message:
        raise NotImplementedError


class LiteLLMClient(LLM):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model

    def complete(self, messages: list[Message], *, tools: list[dict] | None = None,
                 json_mode: bool = False, mock_response: MockFn | None = None) -> Message:
        import litellm  # imported lazily so offline runs never touch the network stack

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = litellm.completion(**kwargs)
        self.usage.calls += 1
        if resp.usage:
            self.usage.prompt_tokens += resp.usage.prompt_tokens or 0
            self.usage.completion_tokens += resp.usage.completion_tokens or 0
        try:
            self.usage.cost_usd += float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:  # pricing unknown for some models; tokens are still tracked
            self.usage.notes.append(f"cost unavailable for {self.model}")
        msg = resp.choices[0].message
        calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments or "{}")}
            for tc in (msg.tool_calls or [])
        ]
        return {"content": msg.content, "tool_calls": calls}


class MockLLM(LLM):
    """Deterministic stand-in: delegates to the caller's rule-based `mock_response`."""

    offline = True
    model = "mock"

    def complete(self, messages: list[Message], *, tools: list[dict] | None = None,
                 json_mode: bool = False, mock_response: MockFn | None = None) -> Message:
        self.usage.calls += 1
        if mock_response is None:
            return {"content": "Offline demo mode: no rule-based answer is available for this request.",
                    "tool_calls": []}
        out = mock_response(messages)
        return {"content": out.get("content"), "tool_calls": out.get("tool_calls", [])}


def _has_key(model: str) -> bool:
    if model.startswith("openai/") or model.startswith("gpt"):
        return bool(os.getenv("OPENAI_API_KEY"))
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def get_llm() -> LLM:
    """Real client when a model + key are configured, otherwise the offline MockLLM."""
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()
    if model == "mock" or not _has_key(model):
        return MockLLM()
    return LiteLLMClient(model)


def assistant_message(resp: Message) -> Message:
    """Convert a `complete` result into an OpenAI-format assistant message for the history."""
    msg: Message = {"role": "assistant", "content": resp.get("content") or ""}
    if resp.get("tool_calls"):
        msg["tool_calls"] = [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
            for tc in resp["tool_calls"]
        ]
    return msg
