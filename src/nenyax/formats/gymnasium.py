"""Gymnasium (Farama) driver: the classic ``reset/step`` API, mapped to step mode.

Also covers anything registered with Gymnasium (MuJoCo, Atari via ale-py, Minigrid, ...).
"""

from __future__ import annotations

from typing import Any

from .._util import to_json
from ..environment import Session, StepEnvironment
from ..registry import Driver
from ..types import JSON, Capabilities, Judgment, Manifest, Mode, Observation, Step, TaskRef


def space_schema(space: Any) -> dict[str, JSON]:
    """JSON Schema for a Gymnasium space (best effort; unknown spaces are described by name)."""
    import gymnasium.spaces as S

    if isinstance(space, S.Discrete):
        return {
            "type": "integer",
            "minimum": int(space.start),
            "maximum": int(space.start + space.n - 1),
        }
    if isinstance(space, S.Box):
        return {
            "type": "array",
            "x-shape": list(space.shape),
            "x-dtype": str(space.dtype),
            "x-low": to_json(space.low),
            "x-high": to_json(space.high),
        }
    if isinstance(space, S.MultiBinary):
        return {
            "type": "array",
            "items": {"type": "integer", "enum": [0, 1]},
            "x-shape": to_json(space.shape),
        }
    if isinstance(space, S.MultiDiscrete):
        return {"type": "array", "items": {"type": "integer"}, "x-nvec": to_json(space.nvec)}
    if isinstance(space, S.Text):
        return {"type": "string", "maxLength": int(space.max_length)}
    if isinstance(space, S.Dict):
        return {
            "type": "object",
            "properties": {k: space_schema(v) for k, v in space.spaces.items()},
        }
    if isinstance(space, S.Tuple):
        return {"type": "array", "prefixItems": [space_schema(s) for s in space.spaces]}
    return {"x-space": repr(space)}


def to_native_action(space: Any, action: JSON) -> Any:
    import gymnasium.spaces as S
    import numpy as np

    if isinstance(space, S.Discrete):
        return int(action)
    if isinstance(space, (S.Box, S.MultiBinary, S.MultiDiscrete)):
        return np.asarray(action, dtype=space.dtype).reshape(space.shape)
    return action


class GymnasiumSession(Session):
    def __init__(self, env: Any, seed: int | None) -> None:
        self.env = env
        self.history: list[Step] = []
        obs, info = env.reset(seed=seed)
        self.observation = self._obs(obs, info)
        self.done = False

    def _obs(self, obs: Any, info: dict) -> Observation:
        text = obs if isinstance(obs, str) else None
        return Observation(data=to_json(obs), text=text, info=to_json(info))

    def act(self, action: JSON) -> Step:
        if self.done:
            raise RuntimeError("episode is over; open a new session")
        native = to_native_action(self.env.action_space, action)
        obs, reward, terminated, truncated, info = self.env.step(native)
        step = Step(
            action=to_json(action),
            observation=self._obs(obs, info),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=to_json(info),
        )
        self.history.append(step)
        self.observation = step.observation
        self.done = step.done
        return step

    def judge(self) -> Judgment:
        return Judgment(
            score=sum(s.reward for s in self.history),
            source="env",
            judge="gymnasium:episode_return",
            details={"steps": len(self.history)},
        )

    def close(self) -> None:
        self.env.close()


class GymnasiumEnvironment(StepEnvironment):
    def __init__(self, env_id: str, make_kwargs: dict[str, Any]) -> None:
        import gymnasium as gym

        self._gym = gym
        self._env_id = env_id
        self._kwargs = make_kwargs
        probe = gym.make(env_id, **make_kwargs)
        try:
            text = isinstance(probe.observation_space, gym.spaces.Text)
            spec = probe.spec
            self.default_max_steps = (spec.max_episode_steps if spec else None) or 10_000
            self.manifest = Manifest(
                id=f"gymnasium:{env_id}",
                name=env_id,
                source_format="gymnasium",
                mode=Mode.STEP,
                version=gym.__version__,
                capabilities=Capabilities(
                    resettable=True,
                    branchable=False,
                    text=text,
                    reward_timing="per_step",
                ),
                action_schema=space_schema(probe.action_space),
                observation_schema=space_schema(probe.observation_space),
            )
            self._action_space = probe.action_space
        finally:
            probe.close()

    def session(self, *, task: TaskRef | None = None, seed: int | None = None) -> GymnasiumSession:
        return GymnasiumSession(self._gym.make(self._env_id, **self._kwargs), seed)

    def random_policy(self, seed: int | None = None):
        space = self._gym.make(self._env_id, **self._kwargs).action_space
        space.seed(seed)
        return lambda observation: to_json(space.sample())


class GymnasiumDriver(Driver):
    name = "gymnasium"
    format = "Gymnasium (Farama Foundation)"
    module = "gymnasium"
    extra = "gymnasium"

    def load(self, target: str, **options: Any) -> GymnasiumEnvironment:
        self.require("gymnasium")
        return GymnasiumEnvironment(target, options)
