"""Policies: whatever chooses actions.

There are exactly two kinds, because environments consume exactly two kinds:

* :class:`ActionPolicy` maps an :class:`~nenyax.types.Observation` to a native action. Step-mode
  environments with non-text observations (CartPole, board games) need this.
* :class:`ChatModel` maps chat messages to an assistant message. LLM environments need this,
  whether they call it themselves (rollout/task mode) or we render observations as text.

Any ``ChatModel`` can drive a text-capable step environment via :func:`chat_policy`, and any
``ChatModel`` (including a plain Python function) can be exposed as an OpenAI-compatible
endpoint for rollout/task environments via :class:`nenyax.server.ModelServer`.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .types import JSON, Message, ModelCall, Observation, extract_tokens


@runtime_checkable
class ActionPolicy(Protocol):
    def __call__(self, observation: Observation) -> JSON: ...


@runtime_checkable
class ChatModel(Protocol):
    def complete(self, messages: Sequence[Message], **params: Any) -> Message: ...


@dataclass
class FunctionModel:
    """Wrap ``fn(messages) -> str | Message | dict`` as a :class:`ChatModel`."""

    fn: Callable[[list[Message]], str | Message | dict]
    name: str = "nenyax-function"

    def complete(self, messages: Sequence[Message], **params: Any) -> Message:
        out = self.fn(list(messages))
        if isinstance(out, Message):
            return out
        if isinstance(out, dict):
            return Message.model_validate({"role": "assistant", **out})
        return Message(role="assistant", content=str(out))


@dataclass
class Endpoint:
    """An OpenAI-compatible chat-completions endpoint (vLLM, SGLang, OpenAI, Together, ...)."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_s: float = 600.0
    default_params: dict[str, Any] = field(default_factory=dict)

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST a raw chat-completions body and return the raw JSON response."""
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:  # surface the server's explanation
            raise RuntimeError(
                f"endpoint {self.base_url} returned {e.code}: {e.read()[:500]!r}"
            ) from e

    def complete(self, messages: Sequence[Message], **params: Any) -> Message:
        body = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            **self.default_params,
            **params,
        }
        data = self.chat(body)
        return Message.model_validate(data["choices"][0]["message"])


@dataclass(frozen=True)
class NativeAgent:
    """Use an agent that ships with the native format (e.g. Harbor's ``oracle`` or ``nop``).

    Only meaningful for environments whose format has built-in agents; used mostly by audits.
    """

    name: str
    config: dict[str, Any] = field(default_factory=dict)


class RecordingModel:
    """Wrap a :class:`ChatModel` and record every call as a :class:`ModelCall`.

    For an :class:`Endpoint` the raw response is kept, so token ids and logprobs survive.
    """

    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.calls: list[ModelCall] = []

    def complete(self, messages: Sequence[Message], **params: Any) -> Message:
        t0 = time.perf_counter()
        if isinstance(self.model, Endpoint):
            body = {
                "model": self.model.model,
                "messages": [m.to_openai() for m in messages],
                **self.model.default_params,
                **params,
            }
            call = ModelCall(request=body)
            try:
                call.response = self.model.chat(body)
            except Exception as e:
                call.error = f"{type(e).__name__}: {e}"
                raise
            finally:
                call.latency_s = time.perf_counter() - t0
                self.calls.append(call)
            extract_tokens(call, call.response)
            return Message.model_validate(call.response["choices"][0]["message"])
        reply = self.model.complete(messages, **params)
        self.calls.append(
            ModelCall(
                request={"messages": [m.to_openai() for m in messages], **params},
                response={"choices": [{"index": 0, "message": reply.to_openai()}]},
                latency_s=time.perf_counter() - t0,
            )
        )
        return reply


class ChatPolicy:
    """Drive a text-capable step environment with a chat model.

    Each observation becomes a user turn and each assistant reply becomes the action. The
    conversation accumulates within an episode; rollout loops call :meth:`reset` between episodes.
    """

    def __init__(self, model: ChatModel, system: str | None = None) -> None:
        self.model = model if isinstance(model, RecordingModel) else RecordingModel(model)
        self.system = system
        self.history: list[Message] = []
        self.reset()

    @property
    def calls(self) -> list[ModelCall]:
        return self.model.calls

    def reset(self) -> None:
        self.history = [Message(role="system", content=self.system)] if self.system else []
        self.model.calls = []

    def __call__(self, observation: Observation) -> JSON:
        if observation.messages:
            self.history.extend(observation.messages)
        else:
            text = (
                observation.text if observation.text is not None else json.dumps(observation.data)
            )
            self.history.append(Message(role="user", content=text))
        reply = self.model.complete(self.history)
        self.history.append(reply)
        return reply.content if isinstance(reply.content, str) else ""


def chat_policy(model: ChatModel, system: str | None = None) -> ChatPolicy:
    return ChatPolicy(model, system)
