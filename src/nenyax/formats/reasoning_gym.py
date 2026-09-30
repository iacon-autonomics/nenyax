"""reasoning-gym driver: procedurally generated reasoning tasks with built-in verifiers.

Each task is a single-turn text episode: the observation is the question, the action is the
answer, and the dataset's own ``score_answer`` is the verifier.
"""

from __future__ import annotations

import re
from typing import Any

from .._util import to_json
from ..environment import Session, StepEnvironment
from ..registry import Driver
from ..types import JSON, Capabilities, Judgment, Manifest, Mode, Observation, Step, TaskRef

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
INSTRUCTION = "Put your final answer inside <answer></answer> tags."


def extract_answer(text: str) -> str:
    """Use the last ``<answer>...</answer>`` span if present, else the whole reply."""
    found = ANSWER_TAG.findall(text or "")
    return (found[-1] if found else text or "").strip()


class ReasoningGymSession(Session):
    def __init__(self, dataset: Any, index: int, judge_name: str) -> None:
        self.dataset = dataset
        self.index = index
        self.entry = dataset[index]
        self.judge_name = judge_name
        self.history: list[Step] = []
        self.done = False
        self.observation = Observation(
            data={"question": self.entry["question"]},
            text=f"{self.entry['question']}\n\n{INSTRUCTION}",
            info={"task_id": str(index)},
        )

    def act(self, action: JSON) -> Step:
        if self.done:
            raise RuntimeError("episode is over; open a new session")
        answer = extract_answer(str(action))
        score = float(self.dataset.score_answer(answer=answer, entry=self.entry))
        step = Step(
            action=action,
            observation=Observation(text="", info={"extracted_answer": answer}),
            reward=score,
            terminated=True,
        )
        self.history.append(step)
        self.done = True
        return step

    def judge(self) -> Judgment:
        score = self.history[-1].reward if self.history else 0.0
        return Judgment(
            score=score,
            passed=score >= 1.0,
            source="verifier",
            judge=self.judge_name,
            details={"expected": to_json(self.entry.get("answer"))},
        )


class ReasoningGymEnvironment(StepEnvironment):
    default_max_steps = 1

    def __init__(self, name: str, size: int = 100, seed: int = 0, **config: Any) -> None:
        import reasoning_gym

        self.dataset = reasoning_gym.create_dataset(name, size=size, seed=seed, **config)
        self.size = size
        version = getattr(reasoning_gym, "__version__", None)
        self._judge = f"reasoning_gym:{name}.score_answer"
        self.manifest = Manifest(
            id=f"reasoning_gym:{name}",
            name=name,
            source_format="reasoning_gym",
            mode=Mode.STEP,
            version=version,
            license="Apache-2.0",
            capabilities=Capabilities(resettable=True, text=True, reward_timing="terminal"),
            action_schema={"type": "string"},
            observation_schema={"type": "string"},
            num_tasks=size,
            extra={"seed": seed, "config": to_json(config)},
        )

    def tasks(self, split: str | None = None, limit: int | None = None) -> list[TaskRef]:
        n = self.size if limit is None else min(limit, self.size)
        return [TaskRef(id=str(i), prompt=self.dataset[i]["question"]) for i in range(n)]

    def _index(self, task: TaskRef | None, seed: int | None) -> int:
        if task is not None:
            return int(task.id)
        return (seed or 0) % self.size

    def session(
        self, *, task: TaskRef | None = None, seed: int | None = None
    ) -> ReasoningGymSession:
        return ReasoningGymSession(self.dataset, self._index(task, seed), self._judge)

    def reference_policy(self, task: TaskRef | None = None):
        dataset = self.dataset
        return lambda obs: f"<answer>{dataset[int(obs.info['task_id'])]['answer']}</answer>"

    def null_policy(self, task: TaskRef | None = None):
        return lambda obs: "<answer></answer>"


class ReasoningGymDriver(Driver):
    name = "reasoning_gym"
    format = "reasoning-gym"
    module = "reasoning_gym"
    extra = "reasoning-gym"

    def load(self, target: str, **options: Any) -> ReasoningGymEnvironment:
        self.require("reasoning_gym")
        return ReasoningGymEnvironment(target, **options)
