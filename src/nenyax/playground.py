"""Play any environment by hand: the operations behind ``nenyax env play`` and the web playground.

A :class:`Playground` wraps one environment and answers small JSON operations::

    describe                 what it is, its mode, action/observation formats, tools
    tasks   {offset, limit}  its tasks (questions, levels, instances), when it has a task set
    reset   {task?, seed?}   start an episode; returns the first observation
    step    {action}         act; returns observation, reward, terminated/truncated, totals
    sample                   a random valid action, when the environment can draw one
    judge                    score the episode so far
    close                    end the episode

Rollout- and task-mode environments call a model themselves, so there you play the model::

    start_rollout {task?, seed?}   run an episode with you as the model (a background thread)
    turn {after?}                  the request waiting for you (messages, tools), or the result
    reply {content?, tool_calls?}  answer it; returns the next turn or the finished result
    abort                          stop the episode

The same operations run locally, on your runner, or in a Nenyax Cloud sandbox (``remote.py``
streams them through the platform), so the playground works wherever the environment does.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ._util import to_json
from .environment import Environment, Session, UnsupportedCapability
from .human import HumanModel
from .types import Mode, Observation, Step, Trajectory

MAX_TEXT = 20_000
#: How long ``turn`` and ``reply`` wait for the environment before answering "still working";
#: kept under the platform's request window so a slow environment never times a request out.
TURN_WAIT_S = 12.0


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
        # Playing as the model (rollout/task mode).
        self.human: HumanModel | None = None
        self.thread: threading.Thread | None = None
        self.result: Trajectory | None = None

    # -- operations -------------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        info = self.env.describe()
        info["playable_as_model"] = self.env.mode in (Mode.ROLLOUT, Mode.TASK)
        if not info["steppable"]:
            info["play_note"] = (
                f"This environment runs in {info['mode']} mode: it talks to a model itself. "
                "Play it as the model: read what it sends, reply or call its tools, and see "
                "how it scores you."
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
        self.abort()
        return {"closed": True}

    # -- playing as the model ---------------------------------------------------------------

    def start_rollout(self, task: str | None = None, seed: int | None = None) -> dict[str, Any]:
        if self.env.mode == Mode.STEP:
            raise ValueError("this environment is played step by step: use reset and step")
        self.abort()
        ref = None
        if task is not None:
            ref = next((t for t in self.env.tasks() if t.id == str(task)), None)
            if ref is None:
                raise ValueError(f"no task {task!r}")
        human = HumanModel()
        self.human, self.result, self.task_id = human, None, ref.id if ref else None

        def play() -> None:
            self.result = self.env.rollout(human, task=ref, seed=seed)

        self.thread = threading.Thread(target=play, name="nenyax-play-as-model", daemon=True)
        self.thread.start()
        return self.turn()

    def turn(self, after: int = 0, wait_s: float = TURN_WAIT_S) -> dict[str, Any]:
        """The request waiting for the person, the finished result, or "still working"."""
        if self.human is None:
            if self.result is not None:
                return {"status": "done", "task": self.task_id, **self._finished(self.result)}
            raise ValueError("start_rollout first")
        human = self.human
        deadline = time.monotonic() + wait_s
        while True:
            waiting = human.wait_for_turn(0.1, after=after)
            if waiting is not None:
                return {"status": "your_turn", "task": self.task_id, "turn": waiting.public()}
            if self.result is not None:  # the environment finished (scored you)
                return {"status": "done", "task": self.task_id, **self._finished(self.result)}
            if time.monotonic() >= deadline:
                return {"status": "working", "task": self.task_id, "turns": len(human.turns)}

    def reply(
        self, content: str | None = None, tool_calls: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        if self.human is None:
            raise ValueError("start_rollout first")
        answered = self.human.answer(content=content, tool_calls=tool_calls)
        # Wait for whatever comes next: the environment's next request, or the episode's end.
        return self.turn(after=answered.index + 1)

    def abort(self) -> dict[str, Any]:
        if self.human is not None:
            self.human.abort()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.human, self.thread = None, None
        return {"aborted": True}

    @staticmethod
    def _finished(traj: Trajectory) -> dict[str, Any]:
        j = traj.judgment
        error = traj.error
        if error and "HumanAborted" in error:
            error = "stopped before the environment finished"
        return {
            "score": j.score if j else None,
            "judgment": j.model_dump(mode="json") if j else None,
            "messages": [m.to_openai() for m in traj.messages][-200:],
            "model_calls": len(traj.model_calls),
            "error": error,
            "duration_s": round(traj.duration_s or 0.0, 2),
        }

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
            "start_rollout": lambda: self.start_rollout(args.get("task"), args.get("seed")),
            "turn": lambda: self.turn(int(args.get("after", 0))),
            "reply": lambda: self.reply(args.get("content"), args.get("tool_calls")),
            "abort": lambda: self.abort(),
        }
        if op not in handlers:
            return {"error": f"unknown operation {op!r}; one of {sorted(handlers)}"}
        try:
            return handlers[op]()
        except Exception as e:  # noqa: BLE001 - report every failure to the person playing
            return {"error": f"{type(e).__name__}: {e}"}
