"""The environment contract.

One method is universal: :meth:`Environment.rollout` runs one episode with a policy and returns a
judged :class:`~nenyax.types.Trajectory`, whatever the environment's native shape.

Step-mode environments additionally offer :meth:`Environment.session` for fine-grained control.
Rollout- and task-mode environments cannot be stepped from outside, so they raise
:class:`UnsupportedCapability` instead of pretending.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from .policy import ActionPolicy, ChatModel, ChatPolicy, Endpoint, FunctionModel, NativeAgent
from .server import ModelServer
from .types import JSON, Judgment, Manifest, Message, Mode, Observation, Step, TaskRef, Trajectory

Policy = ActionPolicy | ChatModel | NativeAgent


class NenyaxError(Exception):
    """Base class for Nenyax errors."""


class UnsupportedCapability(NenyaxError):
    """The environment cannot do what was asked, and says so rather than faking it."""


class PolicyMismatch(NenyaxError):
    """The policy's kind cannot drive this environment."""


class Session(ABC):
    """A live step-mode episode. Use as a context manager."""

    observation: Observation
    done: bool = False

    @abstractmethod
    def act(self, action: JSON) -> Step: ...

    def judge(self) -> Judgment:
        """Episode judgment. Default: the environment's own summed reward."""
        total = sum(s.reward for s in getattr(self, "history", []))
        return Judgment(score=total, source="env")

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Environment(ABC):
    """Base class every driver's environment implements."""

    manifest: Manifest

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def mode(self) -> Mode:
        return self.manifest.mode

    def tasks(self, split: str | None = None, limit: int | None = None) -> list[TaskRef]:
        """Enumerable situations, when the environment has a finite task set."""
        return []

    def session(self, *, task: TaskRef | None = None, seed: int | None = None) -> Session:
        raise UnsupportedCapability(
            f"{self.id} runs in {self.mode.value} mode and cannot be stepped externally; "
            "use rollout() instead"
        )

    @abstractmethod
    def rollout(
        self,
        policy: Policy,
        *,
        task: TaskRef | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> Trajectory: ...

    def reference_policy(self, task: TaskRef | None = None) -> Policy | None:
        """A policy expected to succeed (an oracle), if the format provides one. Used by audits."""
        return None

    def null_policy(self, task: TaskRef | None = None) -> Policy | None:
        """A policy expected to fail (does nothing useful). Used by audits."""
        return None

    def probe_policy(self) -> Policy | None:
        """A cheap chat model that speaks this environment's expected reply format.

        Conformance uses it to prove the model path works end to end (calls are routed through
        the endpoint and recorded) without needing a real model.
        """
        return None

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    def __enter__(self) -> Environment:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id} mode={self.mode.value}>"


def _is_chat_model(policy: object) -> bool:
    return callable(getattr(policy, "complete", None))


class StepEnvironment(Environment):
    """Base for step-mode environments: implement :meth:`session`, get :meth:`rollout` free."""

    default_max_steps: int = 1_000

    @abstractmethod
    def session(self, *, task: TaskRef | None = None, seed: int | None = None) -> Session: ...

    def _resolve(self, policy: Policy) -> ActionPolicy:
        if _is_chat_model(policy) and not isinstance(policy, ChatPolicy):
            if not self.manifest.capabilities.text:
                raise PolicyMismatch(
                    f"{self.id} has non-text observations; pass an ActionPolicy, not a ChatModel"
                )
            return ChatPolicy(policy)  # type: ignore[arg-type]
        if not callable(policy):
            raise PolicyMismatch(f"{policy!r} is neither an ActionPolicy nor a ChatModel")
        return policy  # type: ignore[return-value]

    def rollout(
        self,
        policy: Policy,
        *,
        task: TaskRef | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> Trajectory:
        act = self._resolve(policy)
        if callable(getattr(act, "reset", None)):
            act.reset()  # type: ignore[attr-defined]
        traj = Trajectory(env_id=self.id, mode=self.mode, task=task, seed=seed)
        t0 = time.perf_counter()
        limit = max_steps or self.default_max_steps
        try:
            with self.session(task=task, seed=seed) as sess:
                traj.initial_observation = sess.observation
                obs = sess.observation
                while not sess.done:
                    if len(traj.steps) >= limit:
                        traj.steps[-1].truncated = True
                        break
                    step = sess.act(act(obs))
                    traj.steps.append(step)
                    obs = step.observation
                traj.judgment = sess.judge()
        except Exception as e:
            traj.error = f"{type(e).__name__}: {e}"
            traj.metadata["traceback"] = traceback.format_exc()
        if isinstance(act, ChatPolicy):
            traj.messages = list(act.history)
            traj.model_calls = list(act.calls)
        traj.duration_s = time.perf_counter() - t0
        return traj


class ModelDrivenEnvironment(Environment):
    """Base for rollout/task-mode environments that call a model endpoint themselves.

    Subclasses implement :meth:`_run`, receiving an :class:`Endpoint` to hand to the native
    framework. Nenyax serves that endpoint through a recording :class:`ModelServer`, so every
    model call lands in the trajectory regardless of the framework.
    """

    #: Hostname the environment should use to reach the local model server.
    advertise_host: str | None = None

    @abstractmethod
    def _run(
        self, endpoint: Endpoint, *, task: TaskRef | None, seed: int | None, max_steps: int | None
    ) -> tuple[Judgment, list[Message], dict[str, Any]]:
        """Run one native episode against ``endpoint``; return judgment, transcript, metadata."""

    def _run_native(
        self, agent: NativeAgent, *, task: TaskRef | None, seed: int | None, max_steps: int | None
    ) -> tuple[Judgment, list[Message], dict[str, Any]]:
        """Run one episode with a format-native agent. Override if the format has any."""
        raise UnsupportedCapability(f"{self.id} has no native agent {agent.name!r}")

    @staticmethod
    def _as_chat_model(policy: Policy) -> ChatModel:
        if _is_chat_model(policy):
            return policy  # type: ignore[return-value]
        if callable(policy):
            # A bare callable here is interpreted as fn(messages) -> reply.
            return FunctionModel(policy)  # type: ignore[arg-type]
        raise PolicyMismatch(f"{policy!r} cannot act as a chat model")

    @contextmanager
    def serving(self, policy: Policy):
        """Yield ``(server, session_id, endpoint)`` for one recorded episode."""
        backend = self._as_chat_model(policy)
        server = ModelServer(backend).start()
        try:
            session = server.new_session()
            endpoint = Endpoint(
                base_url=server.base_url(session, host=self.advertise_host),
                model=server.model_name,
            )
            yield server, session, endpoint
        finally:
            server.stop()

    def rollout(
        self,
        policy: Policy,
        *,
        task: TaskRef | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> Trajectory:
        traj = Trajectory(env_id=self.id, mode=self.mode, task=task, seed=seed)
        t0 = time.perf_counter()
        if isinstance(policy, NativeAgent):
            try:
                judgment, messages, meta = self._run_native(
                    policy, task=task, seed=seed, max_steps=max_steps
                )
                traj.judgment, traj.messages = judgment, messages
                traj.metadata.update(meta, native_agent=policy.name)
            except Exception as e:
                traj.error = f"{type(e).__name__}: {e}"
                traj.metadata["traceback"] = traceback.format_exc()
            traj.duration_s = time.perf_counter() - t0
            return traj
        with self.serving(policy) as (server, session, endpoint):
            try:
                judgment, messages, meta = self._run(
                    endpoint, task=task, seed=seed, max_steps=max_steps
                )
                traj.judgment, traj.messages = judgment, messages
                traj.metadata.update(meta)
            except Exception as e:
                traj.error = f"{type(e).__name__}: {e}"
                traj.metadata["traceback"] = traceback.format_exc()
            traj.model_calls = server.pop_calls(session)
        traj.duration_s = time.perf_counter() - t0
        return traj


PolicyFactory = Callable[[], Policy]
