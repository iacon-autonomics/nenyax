"""Port out: serve any Nenyax step-mode environment as a standard OpenEnv server.

Whatever the source format (Gymnasium, reasoning-gym, NeMo Gym, ...), the result speaks the
OpenEnv wire protocol (``/ws``, ``/reset``, ``/step``, ``/state``, ``/schema``), so OpenEnv
clients and the trainers built on them (TRL, SkyRL, Unsloth, ...) can use it unchanged::

    nenyax serve "reasoning_gym:basic_arithmetic?size=50" --port 8000

Actions are ``{"value": <native action>}``; observations carry ``data``, ``text``, ``info`` and,
on the final step, the episode's Nenyax judgment in ``metadata["nenyax_judgment"]``.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import Field

from ..environment import Session, StepEnvironment, UnsupportedCapability
from ..registry import load
from ..types import TaskRef


def build_app(uri: str, *, max_concurrent_envs: int = 8, **options: Any) -> Any:
    """Build a FastAPI app exposing ``uri`` over the OpenEnv protocol."""
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironment
    from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State

    source = load(uri, **options)
    if not isinstance(source, StepEnvironment):
        raise UnsupportedCapability(
            f"{source.id} runs in {source.mode.value} mode; only step-mode environments can be "
            "served as OpenEnv step servers"
        )
    manifest = source.manifest

    class NenyaxAction(Action):
        value: Any = Field(description="Native action for the wrapped environment")

    class NenyaxObservation(Observation):
        data: Any = Field(default=None, description="Native observation (JSON)")
        text: str | None = Field(default=None, description="Text rendering, if available")
        info: dict[str, Any] = Field(default_factory=dict)

    class NenyaxState(State):
        env_id: str = manifest.id
        task_id: str | None = None

    class NenyaxOpenEnv(OpenEnvEnvironment):
        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self) -> None:
            super().__init__()
            self._session: Session | None = None
            self._state = NenyaxState()

        def reset(
            self,
            seed: int | None = None,
            episode_id: str | None = None,
            task_id: str | None = None,
            **kwargs: Any,
        ) -> NenyaxObservation:
            if self._session is not None:
                self._session.close()
            task = TaskRef(id=str(task_id)) if task_id is not None else None
            self._session = source.session(task=task, seed=seed)
            self._state = NenyaxState(episode_id=episode_id or uuid.uuid4().hex, task_id=task_id)
            obs = self._session.observation
            return NenyaxObservation(
                data=obs.data, text=obs.text, info=obs.info, done=self._session.done, reward=None
            )

        def step(
            self, action: NenyaxAction, timeout_s: float | None = None, **kwargs: Any
        ) -> NenyaxObservation:
            if self._session is None:
                raise RuntimeError("call reset() before step()")
            step = self._session.act(action.value)
            self._state.step_count += 1
            metadata: dict[str, Any] = {"terminated": step.terminated, "truncated": step.truncated}
            if step.done:
                metadata["nenyax_judgment"] = self._session.judge().model_dump()
            return NenyaxObservation(
                data=step.observation.data,
                text=step.observation.text,
                info=step.observation.info,
                done=step.done,
                reward=step.reward,
                metadata=metadata,
            )

        @property
        def state(self) -> NenyaxState:
            return self._state

        def get_metadata(self) -> EnvironmentMetadata:
            return EnvironmentMetadata(
                name=manifest.name,
                description=manifest.description or f"{manifest.id} served by Nenyax",
                version=manifest.version,
            )

        def close(self) -> None:
            if self._session is not None:
                self._session.close()
                self._session = None

    return create_app(
        NenyaxOpenEnv,
        NenyaxAction,
        NenyaxObservation,
        env_name=manifest.name,
        max_concurrent_envs=max_concurrent_envs,
        state_cls=NenyaxState,
    )


def serve(uri: str, *, host: str = "127.0.0.1", port: int = 8000, **options: Any) -> None:
    import uvicorn

    uvicorn.run(build_app(uri, **options), host=host, port=port, log_level="warning")
