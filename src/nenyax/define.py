"""Define an environment in a few lines, no class and no framework.

Single-turn (a dataset plus a scorer)::

    env = nenyax.define(
        "capitals",
        tasks=[{"prompt": "Capital of France?", "answer": "Paris"}, ...],
        score=lambda task, reply: float(task["answer"].lower() in reply.lower()),
    )

Multi-turn (functions over your own state)::

    env = nenyax.define(
        "guess",
        reset=lambda seed: {"secret": seed % 10, "tries": 0},
        observe=lambda s: f"Guess 0-9. Tries so far: {s['tries']}",
        step=lambda s, action: step_fn(s, action),        # -> (reward, done)
    )

Either way you get a full Nenyax environment: rollout, conformance, serve, train.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .environment import Session, StepEnvironment
from .types import Capabilities, Judgment, Manifest, Mode, Observation, Step, TaskRef


class _SingleTurnSession(Session):
    def __init__(self, env: DefinedEnvironment, index: int) -> None:
        self.env, self.index, self.task = env, index, env.rows[index]
        self.history: list[Step] = []
        self.done = False
        self.observation = Observation(
            text=str(self.task["prompt"]), data=None, info={"task_id": str(index)}
        )

    def act(self, action: Any) -> Step:
        reply = action if isinstance(action, str) else str(action)
        reward = float(self.env.score_fn(self.task, reply))
        step = Step(action=reply, observation=Observation(text=""), reward=reward, terminated=True)
        self.history.append(step)
        self.done = True
        return step

    def judge(self) -> Judgment:
        score = self.history[-1].reward if self.history else 0.0
        return Judgment(score=score, source="verifier", judge=f"{self.env.manifest.name}.score")


class _MultiTurnSession(Session):
    def __init__(self, env: DefinedEnvironment, seed: int | None) -> None:
        self.env = env
        self.state = env.reset_fn(seed or 0)
        self.history: list[Step] = []
        self.done = False
        self.observation = self._observe()

    def _observe(self) -> Observation:
        seen = self.env.observe_fn(self.state)
        return Observation(
            text=seen if isinstance(seen, str) else None,
            data=None if isinstance(seen, str) else seen,
        )

    def act(self, action: Any) -> Step:
        reward, done = self.env.step_fn(self.state, action)
        self.observation = self._observe()
        step = Step(
            action=action, observation=self.observation, reward=float(reward), terminated=bool(done)
        )
        self.history.append(step)
        self.done = step.terminated
        return step


class DefinedEnvironment(StepEnvironment):
    def __init__(
        self,
        name: str,
        *,
        tasks: Sequence[dict[str, Any]] | None = None,
        score: Callable[[dict[str, Any], str], float] | None = None,
        reset: Callable[[int], Any] | None = None,
        observe: Callable[[Any], Any] | None = None,
        step: Callable[[Any, Any], tuple[float, bool]] | None = None,
        reference: Callable[[dict[str, Any]], str] | None = None,
        max_steps: int = 50,
        text: bool = True,
        description: str | None = None,
    ) -> None:
        single = tasks is not None and score is not None
        multi = reset is not None and observe is not None and step is not None
        if single == multi:
            raise ValueError("give either tasks= and score=, or reset=, observe= and step=")
        self.rows = [dict(t) for t in tasks] if single else []
        for i, row in enumerate(self.rows):
            if "prompt" not in row:
                raise ValueError(f"task {i} has no 'prompt'")
        self.score_fn, self.reset_fn, self.observe_fn, self.step_fn = score, reset, observe, step
        self.reference_fn = reference
        self.default_max_steps = 1 if single else max_steps
        self.manifest = Manifest(
            id=f"defined:{name}",
            name=name,
            source_format="nenyax.define",
            mode=Mode.STEP,
            description=description,
            capabilities=Capabilities(
                resettable=True, text=text, reward_timing="terminal" if single else "per_step"
            ),
            num_tasks=len(self.rows) if single else None,
        )

    def tasks(self, split: str | None = None, limit: int | None = None) -> list[TaskRef]:
        return [
            TaskRef(
                id=str(i),
                prompt=str(row["prompt"]),
                metadata={k: v for k, v in row.items() if k != "prompt"},
            )
            for i, row in enumerate(self.rows)
        ][:limit]

    def session(self, *, task: TaskRef | None = None, seed: int | None = None) -> Session:
        if self.rows:
            index = int(task.id) if task else (seed or 0) % len(self.rows)
            return _SingleTurnSession(self, index)
        return _MultiTurnSession(self, seed)

    def reference_policy(self, task: TaskRef | None = None):
        if self.reference_fn is None or not self.rows:
            return None
        rows, ref = self.rows, self.reference_fn
        return lambda obs: ref(rows[int(obs.info["task_id"])])

    def null_policy(self, task: TaskRef | None = None):
        return (lambda obs: "") if self.rows else None


def define(name: str, **kwargs: Any) -> DefinedEnvironment:
    return DefinedEnvironment(name, **kwargs)
