"""Judges: define what "good" means, independently of the environment.

A judge is anything with ``judge(traj) -> Judgment``. Wrap any environment, from any format, to
re-score its episodes::

    env = nenyax.with_judge(nenyax.load("verifiers:gsm8k"),
                            nenyax.judges.weighted({env_judge: 1.0, Regex(r"<answer>"): 0.1}))

Built in: the environment's own judgment, exact match, regex/format, Python functions, an LLM
judge with a rubric, and weighted combinations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .environment import Environment, Policy
from .policy import ChatModel
from .types import Judgment, Message, TaskRef, Trajectory


@runtime_checkable
class Judge(Protocol):
    def judge(self, traj: Trajectory) -> Judgment: ...


def final_text(traj: Trajectory) -> str:
    """The last assistant message, or the last step's action, as text."""
    for m in reversed(traj.messages):
        if m.role == "assistant" and isinstance(m.content, str):
            return m.content
    if traj.steps:
        action = traj.steps[-1].action
        return action if isinstance(action, str) else json.dumps(action)
    return ""


class EnvJudge:
    """The environment's own judgment, unchanged. Useful inside combinations."""

    name = "env"

    def judge(self, traj: Trajectory) -> Judgment:
        return traj.judgment or Judgment(score=0.0, source="env", details={"missing": True})


class Function:
    """Wrap ``fn(traj) -> float | Judgment``."""

    def __init__(self, fn: Callable[[Trajectory], float | Judgment], name: str | None = None):
        self.fn, self.name = fn, name or getattr(fn, "__name__", "function")

    def judge(self, traj: Trajectory) -> Judgment:
        out = self.fn(traj)
        if isinstance(out, Judgment):
            return out
        return Judgment(score=float(out), source="verifier", judge=f"function:{self.name}")


class ExactMatch:
    """1.0 if the final answer equals the expected one (from the task, or a mapping)."""

    name = "exact_match"

    def __init__(
        self,
        expected: Callable[[TaskRef | None], str] | None = None,
        extract: str | None = r"<answer>(.*?)</answer>",
        normalize: bool = True,
    ):
        self.expected = expected or (lambda task: task.metadata.get("answer") if task else "")
        self.pattern = re.compile(extract, re.S) if extract else None
        self.normalize = normalize

    def _clean(self, s: str) -> str:
        return " ".join(s.split()).lower() if self.normalize else s

    def judge(self, traj: Trajectory) -> Judgment:
        text = final_text(traj)
        if self.pattern:
            found = self.pattern.findall(text)
            text = found[-1] if found else ""
        want = str(self.expected(traj.task))
        ok = self._clean(text) == self._clean(want)
        return Judgment(
            score=float(ok),
            passed=ok,
            source="verifier",
            judge=self.name,
            details={"got": text[:200], "expected": want[:200]},
        )


class Regex:
    """Format reward: 1.0 if the final text matches ``pattern``."""

    def __init__(self, pattern: str, flags: int = re.S) -> None:
        self.pattern = re.compile(pattern, flags)
        self.name = f"regex:{pattern}"

    def judge(self, traj: Trajectory) -> Judgment:
        ok = bool(self.pattern.search(final_text(traj)))
        return Judgment(score=float(ok), passed=ok, source="verifier", judge=self.name)


_RUBRIC_PROMPT = """You are grading an AI assistant's work.

Rubric:
{rubric}

Task:
{task}

Assistant's final answer:
{answer}

Reply with only a JSON object: {{"score": <number from 0 to 1>, "reason": "<one sentence>"}}"""


class LLMJudge:
    """Score with a model and a plain-language rubric (the domain expert's path)."""

    def __init__(self, model: ChatModel, rubric: str, name: str = "llm_judge") -> None:
        self.model, self.rubric, self.name = model, rubric, name

    def judge(self, traj: Trajectory) -> Judgment:
        task = (traj.task.prompt if traj.task and traj.task.prompt else None) or next(
            (m.content for m in traj.messages if m.role == "user" and isinstance(m.content, str)),
            "",
        )
        prompt = _RUBRIC_PROMPT.format(rubric=self.rubric, task=task, answer=final_text(traj))
        reply = self.model.complete([Message(role="user", content=prompt)]).content or ""
        match = re.search(r"\{.*\}", reply, re.S)
        try:
            parsed = json.loads(match.group(0)) if match else {}
            score = min(1.0, max(0.0, float(parsed.get("score", 0.0))))
        except (ValueError, TypeError):
            parsed, score = {}, 0.0
        return Judgment(
            score=score,
            source="rubric",
            judge=self.name,
            details={"reason": parsed.get("reason"), "raw": reply[:500]},
        )


class Weighted:
    """Weighted sum of judges; every component is kept in ``Judgment.components``."""

    def __init__(self, parts: dict[Any, float]) -> None:
        self.parts = parts
        self.name = "weighted"

    def judge(self, traj: Trajectory) -> Judgment:
        components, total = {}, 0.0
        for judge, weight in self.parts.items():
            j = judge.judge(traj)
            components[getattr(judge, "name", type(judge).__name__)] = j.score
            total += weight * j.score
        return Judgment(score=total, source="verifier", judge=self.name, components=components)


def weighted(parts: dict[Any, float]) -> Weighted:
    return Weighted(parts)


class JudgedEnvironment(Environment):
    """Any environment, re-scored by a judge. The original judgment is kept in details."""

    def __init__(self, inner: Environment, judge: Judge) -> None:
        self.inner, self.judge_ = inner, judge
        self.manifest = inner.manifest.model_copy(update={"id": f"{inner.id}+judge"})

    def tasks(self, split=None, limit=None):
        return self.inner.tasks(split, limit)

    def session(self, *, task=None, seed=None):
        return self.inner.session(task=task, seed=seed)

    def rollout(self, policy: Policy, *, task=None, seed=None, max_steps=None) -> Trajectory:
        traj = self.inner.rollout(policy, task=task, seed=seed, max_steps=max_steps)
        if traj.error:
            return traj
        original = traj.judgment
        judgment = self.judge_.judge(traj)
        if original is not None:
            judgment.details.setdefault("env_judgment", original.model_dump())
        traj.judgment = judgment
        return traj

    def reference_policy(self, task=None):
        return self.inner.reference_policy(task)

    def null_policy(self, task=None):
        return self.inner.null_policy(task)

    def probe_policy(self):
        return self.inner.probe_policy()

    def close(self) -> None:
        self.inner.close()


def with_judge(env: Environment, judge: Judge) -> JudgedEnvironment:
    return JudgedEnvironment(env, judge)
