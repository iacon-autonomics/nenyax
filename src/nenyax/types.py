"""The Nenyax contract: the data every environment exchanges, whatever its native format.

Everything here is plain, serializable data. Drivers translate native objects into these
types; trainers and tools consume only these types.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JSON = Any
"""A JSON-serializable value."""

PROTOCOL_VERSION = "0.1"
"""Version of the Nenyax contract (these types). Minor versions only add optional fields;
removals or meaning changes bump it. Every Manifest and Trajectory records the version it used."""


class Mode(StrEnum):
    """How an environment is driven. Every native format maps to exactly one mode."""

    STEP = "step"
    """The caller drives each step: open a session, observe, act, repeat (Gymnasium, OpenEnv)."""

    ROLLOUT = "rollout"
    """The environment runs the whole episode against a model endpoint (Verifiers, NeMo Gym)."""

    TASK = "task"
    """A sandboxed task: an agent works inside it and tests grade the end state (Harbor)."""


class Risk(StrEnum):
    NONE = "none"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class Isolation(StrEnum):
    """How strongly a sandbox separates untrusted actions from the host, weakest first."""

    NONE = "none"
    """No sandbox needed: pure functions, simulated state (math, verify-style environments)."""
    PROCESS = "process"
    """A separate OS process with resource limits; shares the host filesystem and network."""
    JAIL = "jail"
    """Namespaced process (bubblewrap, nsjail): private filesystem view, optional no-network."""
    CONTAINER = "container"
    """Container (Docker, Podman, gVisor): own filesystem and network, shared or proxied kernel."""
    MICROVM = "microvm"
    """Hardware-virtualized guest kernel (Firecracker, Kata, libkrun)."""

    @property
    def rank(self) -> int:
        return list(Isolation).index(self)

    def satisfies(self, required: Isolation | str) -> bool:
        return self.rank >= Isolation(required).rank


class Capabilities(BaseModel):
    """What an environment can honestly do. Drivers must never over-claim."""

    model_config = ConfigDict(extra="forbid")

    resettable: bool = True
    """The same situation can be recreated from a seed or task id."""
    branchable: bool = False
    """A session can be forked mid-episode to try alternatives."""
    max_parallel: int | None = None
    """Upper bound on concurrent sessions; None means unknown or unbounded."""
    participants: int = 1
    text: bool = False
    """Observations can be rendered as text and actions accepted as text (LLM-drivable)."""
    reward_timing: Literal["per_step", "terminal"] = "terminal"
    real_world: bool = False
    risk: Risk = Risk.NONE
    requires: list[Literal["docker", "network", "gpu", "model_endpoint", "api_key"]] = Field(
        default_factory=list
    )
    isolation: Isolation = Isolation.NONE
    """Minimum isolation this environment's actions need when run locally."""
    token_capture: Literal["none", "proxy", "native"] = "none"
    """How model tokens are captured: not at all, by the Nenyax recording proxy, or natively."""


class Manifest(BaseModel):
    """Identity and contract of a loaded environment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    """Canonical URI, e.g. ``gymnasium:CartPole-v1``."""
    protocol: str = PROTOCOL_VERSION
    name: str
    source_format: str
    """Native format the driver translated from, e.g. ``verifiers`` or ``harbor``."""
    mode: Mode
    capabilities: Capabilities = Field(default_factory=Capabilities)
    version: str | None = None
    license: str | None = None
    description: str | None = None
    action_schema: dict[str, JSON] | None = None
    observation_schema: dict[str, JSON] | None = None
    num_tasks: int | None = None
    extra: dict[str, JSON] = Field(default_factory=dict)


class TaskRef(BaseModel):
    """A pointer to one concrete situation in an environment."""

    id: str
    split: str | None = None
    prompt: str | None = None
    metadata: dict[str, JSON] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str | None = None
    name: str
    arguments: dict[str, JSON] | str = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _unwrap_openai(cls, data: Any) -> Any:
        """Accept OpenAI's wire shape ``{"type": "function", "function": {...}}``."""
        if isinstance(data, dict) and isinstance(data.get("function"), dict):
            fn = data["function"]
            return {
                "id": data.get("id"),
                "name": fn.get("name"),
                "arguments": fn.get("arguments", {}),
            }
        return data

    def to_openai(self) -> dict[str, JSON]:
        args = self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments)
        return {
            "id": self.id or f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": self.name, "arguments": args},
        }


class Message(BaseModel):
    """A chat message; parses and emits OpenAI's wire shape, the lingua franca of LLM envs."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[JSON] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, JSON]:
        """Serialize for an OpenAI-compatible chat-completions request or response."""
        out = self.model_dump(exclude_none=True, exclude={"tool_calls"})
        out.setdefault("content", None)
        if self.tool_calls:
            out["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return out


class Observation(BaseModel):
    """What the acting policy is allowed to see."""

    data: JSON = None
    """Native observation, converted to JSON-compatible values."""
    text: str | None = None
    """Text rendering, present when ``capabilities.text`` is true."""
    messages: list[Message] | None = None
    """Chat-shaped rendering, when the environment is conversational."""
    info: dict[str, JSON] = Field(default_factory=dict)


class Step(BaseModel):
    """One transition in a step-mode episode."""

    action: JSON
    observation: Observation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, JSON] = Field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class ModelCall(BaseModel):
    """One request/response pair observed between an environment and a model endpoint."""

    request: dict[str, JSON]
    response: dict[str, JSON] | None = None
    error: str | None = None
    latency_s: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    completion_logprobs: list[float] | None = None


def extract_tokens(call: ModelCall, response: dict[str, Any]) -> None:
    """Copy token ids and logprobs from a vLLM/SGLang/OpenAI-style response, when present."""
    choice = (response.get("choices") or [{}])[0]
    call.prompt_token_ids = response.get("prompt_token_ids") or choice.get("prompt_token_ids")
    call.completion_token_ids = choice.get("token_ids")
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
        call.completion_logprobs = [t.get("logprob", 0.0) for t in logprobs["content"]]


class Judgment(BaseModel):
    """How good an episode was, and on whose authority."""

    score: float
    """Primary scalar reward for the whole episode."""
    passed: bool | None = None
    components: dict[str, float] = Field(default_factory=dict)
    source: Literal["env", "verifier", "rubric", "tests", "human", "external"] = "env"
    judge: str | None = None
    """Identifier and version of whatever produced the score."""
    details: dict[str, JSON] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"score must be finite, got {v!r}")
        return v


class Trajectory(BaseModel):
    """The complete, judged record of one episode. The unit trainers consume."""

    env_id: str
    mode: Mode
    protocol: str = PROTOCOL_VERSION
    task: TaskRef | None = None
    seed: int | None = None
    initial_observation: Observation | None = None
    steps: list[Step] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    model_calls: list[ModelCall] = Field(default_factory=list)
    judgment: Judgment | None = None
    error: str | None = None
    started_at: float = Field(default_factory=time.time)
    duration_s: float | None = None
    metadata: dict[str, JSON] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.judgment is not None

    @property
    def score(self) -> float | None:
        return self.judgment.score if self.judgment else None
