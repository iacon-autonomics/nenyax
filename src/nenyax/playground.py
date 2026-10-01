"""Play any environment by hand: the operations behind ``nenyax env play`` and the web playground.

A :class:`Playground` wraps one environment and answers small JSON operations::

    describe                 what it is, its mode, action/observation formats, tools
    tasks   {offset, limit}  its tasks (questions, levels, instances), when it has a task set
    reset   {task?, seed?}   start an episode; returns the first observation
    step    {action}         act; returns observation, reward, terminated/truncated, totals
    sample                   a random valid action, when the environment can draw one
    judge                    score the episode so far
    close                    end the episode

The same operations run locally, on your runner, or in a Nenyax Cloud sandbox (``remote.py``
streams them through the platform), so the playground works wherever the environment does.
"""

from __future__ import annotations

from typing import Any

from ._util import to_json
from .environment import Environment, Session, UnsupportedCapability
from .types import Observation, Step

MAX_TEXT = 20_000


def _observation(obs: Observation) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if obs.text:
        out["text"] = obs.text[:MAX_TEXT]
    if obs.data is not None:
        out["data"] = to_json(obs.data)
    if obs.messages:
        out["messages"] = [m.model_dump(mode="json", exclude_none=True) for m in obs.messages]
    if obs.info:
        out["info"] = to_json(obs.info)
    return out


class Playground:
    def __init__(self, env: Environment) -> None:
        self.env = env
        self.session: Session | None = None
        self.steps: list[Step] = []
        self.task_id: str | None = None

    # -- operations -------------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        info = self.env.describe()
        if not info["steppable"]:
            info["play_note"] = (
                f"This environment runs in {info['mode']} mode: it drives a model itself, so it "
                "can't be stepped by hand yet. Browse its tasks, or run it with a model."
            )
        return info

    def tasks(self, offset: int = 0, limit: int = 50, split: str | None = None) -> dict[str, Any]:
        refs = self.env.tasks(split=split, limit=offset + limit) or []
        page = refs[offset : offset + limit]
        return {
            "total": self.env.manifest.num_tasks or len(refs),
            "offset": offset,
            "items": [
                {
                    "id": t.id,
                    "split": t.split,
                    "prompt": (t.prompt or "")[:MAX_TEXT],
                    "metadata": to_json(t.metadata),
                }
                for t in page
            ],
        }

    def reset(self, task: str | None = None, seed: int | None = None) -> dict[str, Any]:
        self.close()
        ref = None
        if task is not None:
            ref = next((t for t in self.env.tasks() if t.id == str(task)), None)
            if ref is None:
                raise ValueError(f"no task {task!r}")
        try:
            try:
                self.session = self.env.session(task=ref, seed=seed, render=True)  # type: ignore[call-arg]
            except TypeError:  # drivers without a render option
                self.session = self.env.session(task=ref, seed=seed)
        except UnsupportedCapability as e:
            raise ValueError(str(e)) from None
        self.steps, self.task_id = [], ref.id if ref else None
        return self._state(first=True)

    def step(self, action: Any) -> dict[str, Any]:
        if self.session is None:
            raise ValueError("reset first")
        if self.session.done:
            raise ValueError("the episode is over; reset to play again")
        step = self.session.act(action)
        self.steps.append(step)
        return {
            **self._state(),
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "step_info": to_json(step.info),
        }

    def sample(self) -> dict[str, Any]:
        if self.session is None:
            raise ValueError("reset first")
        return {"action": self.session.sample_action()}

    def judge(self) -> dict[str, Any]:
        if self.session is None:
            raise ValueError("reset first")
        j = self.session.judge()
        return j.model_dump(mode="json")

    def close(self) -> dict[str, Any]:
        if self.session is not None:
            self.session.close()
        self.session = None
        return {"closed": True}

    # -- helpers ----------------------------------------------------------------------------

    def _state(self, first: bool = False) -> dict[str, Any]:
        s = self.session
        assert s is not None
        return {
            "task": self.task_id,
            "observation": _observation(s.observation),
            "done": s.done,
            "steps": len(self.steps),
            "total_reward": sum(x.reward for x in self.steps),
            "legal_actions": s.legal_actions(),
            "render": s.render(),
            "first": first,
        }

    def handle(self, op: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one operation; errors come back as ``{"error": ...}`` instead of raising."""
        args = args or {}
        handlers = {
            "describe": lambda: self.describe(),
            "tasks": lambda: self.tasks(
                int(args.get("offset", 0)), int(args.get("limit", 50)), args.get("split")
            ),
            "reset": lambda: self.reset(args.get("task"), args.get("seed")),
            "step": lambda: self.step(args.get("action")),
            "sample": lambda: self.sample(),
            "judge": lambda: self.judge(),
            "close": lambda: self.close(),
        }
        if op not in handlers:
            return {"error": f"unknown operation {op!r}; one of {sorted(handlers)}"}
        try:
            return handlers[op]()
        except Exception as e:  # noqa: BLE001 - report every failure to the person playing
            return {"error": f"{type(e).__name__}: {e}"}
