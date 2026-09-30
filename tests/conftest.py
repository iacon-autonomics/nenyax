"""Tiny in-memory environments that exercise every code path without third-party packages."""

from __future__ import annotations

import json
from typing import Any

import pytest

from nenyax import (
    Capabilities,
    Endpoint,
    Judgment,
    Manifest,
    Message,
    Mode,
    ModelDrivenEnvironment,
    Observation,
    Session,
    Step,
    StepEnvironment,
    TaskRef,
)


class CountdownSession(Session):
    """Say the target number; one step; reward 1 if correct."""

    def __init__(self, target: int, lenient: bool) -> None:
        self.target, self.lenient = target, lenient
        self.history: list[Step] = []
        self.done = False
        self.observation = Observation(
            data={"target": target}, text=f"Say {target}", info={"target": target}
        )

    def act(self, action: Any) -> Step:
        ok = self.lenient or str(action).strip() == str(self.target)
        step = Step(
            action=action, observation=Observation(text="done"), reward=float(ok), terminated=True
        )
        self.history.append(step)
        self.done = True
        return step

    def judge(self) -> Judgment:
        return Judgment(score=self.history[-1].reward, source="verifier", judge="countdown@1")


class CountdownEnv(StepEnvironment):
    def __init__(self, *, lenient: bool = False, text: bool = True) -> None:
        self.lenient = lenient
        self.manifest = Manifest(
            id="toy:countdown",
            name="countdown",
            source_format="toy",
            mode=Mode.STEP,
            capabilities=Capabilities(text=text),
            num_tasks=5,
        )

    def tasks(self, split=None, limit=None):
        return [TaskRef(id=str(i)) for i in range(limit or 5)]

    def session(self, *, task=None, seed=None):
        return CountdownSession(int(task.id) if task else (seed or 0), self.lenient)

    def reference_policy(self, task=None):
        return lambda obs: str(obs.info["target"])

    def null_policy(self, task=None):
        return lambda obs: "nothing"


class EchoJudgeEnv(ModelDrivenEnvironment):
    """A rollout-mode env: it calls the model itself, then scores the reply."""

    def __init__(self, *, bypass: bool = False) -> None:
        self.bypass = bypass
        self.manifest = Manifest(
            id="toy:echo-judge",
            name="echo-judge",
            source_format="toy",
            mode=Mode.ROLLOUT,
            capabilities=Capabilities(requires=["model_endpoint"], token_capture="proxy"),
            num_tasks=3,
        )

    def tasks(self, split=None, limit=None):
        return [TaskRef(id=w) for w in ["apple", "river", "stone"][: limit or 3]]

    def _run(self, endpoint: Endpoint, *, task, seed, max_steps):
        word = task.id if task else "apple"
        messages = [Message(role="user", content=f"Repeat exactly: {word}")]
        if self.bypass:
            reply = Message(role="assistant", content=word)
        else:
            reply = endpoint.complete(messages)
        score = float((reply.content or "").strip() == word)
        return Judgment(score=score, source="verifier"), [*messages, reply], {"word": word}

    def reference_policy(self, task=None):
        return lambda messages: messages[-1].content.split(": ", 1)[1]

    def null_policy(self, task=None):
        return lambda messages: ""


@pytest.fixture
def countdown() -> CountdownEnv:
    return CountdownEnv()


@pytest.fixture
def echo_env() -> EchoJudgeEnv:
    return EchoJudgeEnv()


def dumps(x: Any) -> str:
    return json.dumps(x, sort_keys=True)
