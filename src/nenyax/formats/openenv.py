"""OpenEnv driver: any OpenEnv server (local, Docker, or a Hugging Face Space), no per-env code.

Uses OpenEnv's ``GenericEnvClient`` (raw JSON actions over the ``/ws`` protocol) and the
server's ``/schema`` endpoint, so one driver covers every OpenEnv environment::

    nenyax.load("openenv:https://openenv-openspiel-env.hf.space")
    nenyax.load("openenv:http://localhost:8000?reset={\\"seed\\":1}")

MCP-style environments are reachable too: actions shaped like
``{"type": "call_tool", "tool_name": ..., "arguments": {...}}``.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from .._util import to_json
from ..environment import Session, StepEnvironment
from ..registry import Driver
from ..types import JSON, Capabilities, Judgment, Manifest, Mode, Observation, Step, TaskRef


def _get_json(url: str, timeout: float = 30.0) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _observation(obs: Any, reward: float | None, metadata: Any) -> Observation:
    data = to_json(obs)
    text = None
    if isinstance(data, dict):
        for key in ("text", "message", "prompt", "content", "stdout"):
            if isinstance(data.get(key), str):
                text = data[key]
                break
    return Observation(data=data, text=text, info={"metadata": to_json(metadata or {})})


class OpenEnvSession(Session):
    def __init__(
        self, client: Any, reset_kwargs: dict[str, Any], text_action: str | None = None
    ) -> None:
        self.client = client
        self.text_action = text_action
        self.history: list[Step] = []
        result = client.reset(**reset_kwargs)
        self.observation = _observation(result.observation, result.reward, result.metadata)
        self.done = bool(result.done)

    def act(self, action: JSON) -> Step:
        if self.done:
            raise RuntimeError("episode is over; open a new session")
        if self.text_action and isinstance(action, str):
            action = {self.text_action: action}
        result = self.client.step(action)
        reward_missing = result.reward is None
        step = Step(
            action=to_json(action),
            observation=_observation(result.observation, result.reward, result.metadata),
            reward=0.0 if reward_missing else float(result.reward),
            terminated=bool(result.done),
            info={"reward_missing": True} if reward_missing else {},
        )
        self.history.append(step)
        self.observation = step.observation
        self.done = step.done
        return step

    def state(self) -> JSON:
        """Server-side state. Privileged: may contain answers, never show it to the policy."""
        return to_json(self.client.state())

    def judge(self) -> Judgment:
        return Judgment(
            score=sum(s.reward for s in self.history),
            source="env",
            judge="openenv:episode_return",
            details={"steps": len(self.history)},
        )

    def close(self) -> None:
        self.client.close()


class OpenEnvEnvironment(StepEnvironment):
    def __init__(
        self,
        base_url: str,
        *,
        reset: dict[str, Any] | None = None,
        seeded: bool = False,
        text_action: str | None = None,
        message_timeout_s: float = 300.0,
    ) -> None:
        """
        Args:
            base_url: OpenEnv server URL (``http(s)://...``).
            reset: keyword arguments passed to every ``reset()``.
            seeded: set True only if this server honours ``reset(seed=...)``; Nenyax then passes
                the rollout seed through and reports the environment as resettable.
            text_action: name of the action field that takes free text (e.g. ``"message"`` for
                Wordle, ``"value"`` for Nenyax-exported servers). Makes the environment
                drivable by chat models: text replies are wrapped as ``{text_action: reply}``.
        """
        self.base_url = base_url.rstrip("/")
        self.reset_kwargs = dict(reset or {})
        self.seeded = seeded
        self.text_action = text_action
        self.message_timeout_s = message_timeout_s
        schema = _get_json(f"{self.base_url}/schema") or {}
        meta = _get_json(f"{self.base_url}/metadata") or {}
        name = meta.get("name") or self.base_url.split("//", 1)[-1]
        self.manifest = Manifest(
            id=f"openenv:{self.base_url}",
            name=name,
            source_format="openenv",
            mode=Mode.STEP,
            version=meta.get("version"),
            description=meta.get("description"),
            capabilities=Capabilities(
                resettable=seeded,
                text=text_action is not None,
                reward_timing="per_step",
                requires=["network"],
            ),
            action_schema=schema.get("action"),
            observation_schema=schema.get("observation"),
            extra={"state_schema": schema.get("state")} if schema.get("state") else {},
        )

    def session(self, *, task: TaskRef | None = None, seed: int | None = None) -> OpenEnvSession:
        from openenv import GenericEnvClient

        kwargs = dict(self.reset_kwargs)
        if self.seeded and seed is not None:
            kwargs.setdefault("seed", seed)
        client = GenericEnvClient(
            base_url=self.base_url, message_timeout_s=self.message_timeout_s
        ).sync()
        client.__enter__()
        try:
            return OpenEnvSession(client, kwargs, self.text_action)
        except Exception:
            client.close()
            raise


class OpenEnvDriver(Driver):
    name = "openenv"
    format = "OpenEnv (Meta PyTorch / Hugging Face)"
    module = "openenv"
    extra = "openenv"

    def load(self, target: str, **options: Any) -> OpenEnvEnvironment:
        self.require("openenv")
        if not target.startswith(("http://", "https://")):
            target = "https://" + target
        return OpenEnvEnvironment(target, **options)
