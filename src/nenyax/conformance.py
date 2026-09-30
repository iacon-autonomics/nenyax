"""Conformance: measure how far an environment really gets, instead of claiming plug-and-play.

Tiers are cumulative. An environment reaches a tier only if every check at that tier and below
passes::

    connectable  loads, manifest is valid, an episode can start
    evaluable    a full episode completes with a finite judgment
    trainable    episodes are reproducible and emit a learning signal a trainer can use
    audited      the verifier discriminates: a reference policy beats a null policy

A check that cannot run (e.g. no reference policy exists) is ``skipped`` and blocks its tier:
an unverifiable verifier is not an audited one.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

from .environment import Environment, Policy, StepEnvironment
from .policy import FunctionModel
from .types import Manifest, Mode, TaskRef, Trajectory


class Tier(IntEnum):
    NONE = 0
    CONNECTABLE = 1
    EVALUABLE = 2
    TRAINABLE = 3
    AUDITED = 4

    @property
    def label(self) -> str:
        return self.name.lower()


Status = Literal["pass", "fail", "skip"]


@dataclass
class Check:
    name: str
    tier: Tier
    status: Status
    detail: str = ""
    duration_s: float = 0.0


@dataclass
class Report:
    env_id: str
    mode: str
    checks: list[Check] = field(default_factory=list)

    @property
    def tier(self) -> Tier:
        reached = Tier.NONE
        for tier in (Tier.CONNECTABLE, Tier.EVALUABLE, Tier.TRAINABLE, Tier.AUDITED):
            at = [c for c in self.checks if c.tier == tier]
            if at and all(c.status == "pass" for c in at):
                reached = tier
            else:
                break
        return reached

    def to_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "mode": self.mode,
            "tier": self.tier.label,
            "checks": [c.__dict__ | {"tier": c.tier.label} for c in self.checks],
        }

    def render(self) -> str:
        icon = {"pass": "✔", "fail": "✘", "skip": "–"}
        lines = [f"{self.env_id}  [{self.mode}]  →  tier: {self.tier.label.upper()}"]
        for c in self.checks:
            timing = f"{c.duration_s:6.2f}s"
            lines.append(f"  {icon[c.status]} {c.tier.label:<11} {c.name:<26} {timing}  {c.detail}")
        return "\n".join(lines)


def _default_probe(env: Environment) -> Policy:
    """A policy that exercises plumbing without needing to be good."""
    ref = env.reference_policy()
    if ref is not None:
        return ref
    random_policy = getattr(env, "random_policy", None)
    if callable(random_policy):
        return random_policy(seed=0)
    if env.manifest.capabilities.text or env.mode is not Mode.STEP:
        return FunctionModel(lambda messages: "I don't know.")
    raise ValueError(f"no probe policy available for {env.id}; pass one explicitly")


def _signal(traj: Trajectory, mode: Mode) -> str | None:
    """Return an error if the trajectory lacks what a trainer needs, else None."""
    if traj.judgment is None or not math.isfinite(traj.judgment.score):
        return "no finite judgment"
    if mode is Mode.STEP:
        if not traj.steps:
            return "no steps recorded"
        return None
    if not traj.model_calls:
        return "no model calls recorded; the environment bypassed the endpoint"
    if not any(c.response for c in traj.model_calls):
        return "model calls recorded without responses"
    return None


def check(
    env: Environment,
    *,
    probe: Policy | None = None,
    reference: Policy | None = None,
    null: Policy | None = None,
    tasks: list[TaskRef] | None = None,
    samples: int = 3,
    on_check: Callable[[Check], None] | None = None,
) -> Report:
    """Run the conformance suite against a loaded environment."""
    report = Report(env_id=env.id, mode=env.mode.value)

    def record(name: str, tier: Tier, fn: Callable[[], str | None], *, skip: str | None = None):
        if skip is not None:
            c = Check(name, tier, "skip", skip)
        else:
            t0 = time.perf_counter()
            try:
                err = fn()
                c = Check(name, tier, "fail" if err else "pass", err or "")
            except Exception as e:
                c = Check(name, tier, "fail", f"{type(e).__name__}: {e}")
            c.duration_s = time.perf_counter() - t0
        report.checks.append(c)
        if on_check:
            on_check(c)
        return c

    # -- connectable ---------------------------------------------------------------------------
    def _manifest() -> str | None:
        Manifest.model_validate(env.manifest.model_dump())
        return None

    record("manifest", Tier.CONNECTABLE, _manifest)

    task_list = tasks if tasks is not None else env.tasks(limit=samples)
    chosen = task_list[:samples] if task_list else [None] * samples

    if isinstance(env, StepEnvironment):

        def _open() -> str | None:
            with env.session(task=chosen[0], seed=0) as s:
                return None if s.observation is not None else "session has no observation"

        record("session_opens", Tier.CONNECTABLE, _open)
    else:
        record(
            "tasks_listed",
            Tier.CONNECTABLE,
            lambda: None if task_list or env.manifest.num_tasks != 0 else "no tasks",
        )

    # -- evaluable -----------------------------------------------------------------------------
    # A fresh probe per episode keeps stateful or seeded policies deterministic.
    def make_probe() -> Policy:
        return probe if probe is not None else _default_probe(env)

    first: list[Trajectory] = []

    def _evaluate() -> str | None:
        for i, task in enumerate(chosen):
            traj = env.rollout(make_probe(), task=task, seed=i)
            first.append(traj)
            if traj.error:
                return traj.error
            if traj.judgment is None:
                return "rollout produced no judgment"
        return None

    ev = record("rollout_completes", Tier.EVALUABLE, _evaluate)

    # -- trainable -----------------------------------------------------------------------------
    blocked = None if ev.status == "pass" else "rollout did not complete"

    def _reproducible() -> str | None:
        again = env.rollout(make_probe(), task=chosen[0], seed=0)
        a, b = first[0], again
        if (
            a.initial_observation
            and b.initial_observation
            and (a.initial_observation.data != b.initial_observation.data)
        ):
            return "same seed produced a different initial observation"
        if env.mode is Mode.STEP and a.score != b.score:
            return f"same seed and policy scored {a.score} then {b.score}"
        return None

    # Group-based RL (GRPO and friends) samples the same situation many times, so a trainable
    # environment must be able to recreate one. Environments that don't claim it are skipped.
    not_resettable = (
        None
        if env.manifest.capabilities.resettable
        else "environment does not declare seeded reset"
    )
    record("reproducible", Tier.TRAINABLE, _reproducible, skip=blocked or not_resettable)

    def _learning_signal() -> str | None:
        if env.mode is Mode.STEP:
            return next((e for e in (_signal(t, env.mode) for t in first) if e), None)
        # Model-driven: prove the model path itself, with a chat-model probe (a reference
        # policy may be a native agent that never calls a model).
        model_probe = env.probe_policy() or FunctionModel(lambda messages: "I don't know.")
        traj = env.rollout(model_probe, task=chosen[0], seed=0)
        return traj.error or _signal(traj, env.mode)

    record("learning_signal", Tier.TRAINABLE, _learning_signal, skip=blocked)

    # -- audited -------------------------------------------------------------------------------
    # Reference policies may be task-specific (an oracle knows each task's answer).
    def ref_for(task: TaskRef | None) -> Policy | None:
        return reference if reference is not None else env.reference_policy(task)

    def nul_for(task: TaskRef | None) -> Policy | None:
        return null if null is not None else env.null_policy(task)

    have_both = ref_for(chosen[0]) is not None and nul_for(chosen[0]) is not None
    audit_skip = blocked or (None if have_both else "no reference/null policy to test the verifier")

    def _discriminates() -> str | None:
        ref_scores, nul_scores = [], []
        for i, task in enumerate(chosen):
            r = env.rollout(ref_for(task), task=task, seed=i)  # type: ignore[arg-type]
            n = env.rollout(nul_for(task), task=task, seed=i)  # type: ignore[arg-type]
            if r.error or n.error:
                return r.error or n.error
            ref_scores.append(r.score or 0.0)
            nul_scores.append(n.score or 0.0)
        bad = [i for i, (r, n) in enumerate(zip(ref_scores, nul_scores, strict=True)) if r <= n]
        if bad:
            return (
                f"reference did not beat null on tasks {bad} (ref={ref_scores}, null={nul_scores})"
            )
        return None

    record("verifier_discriminates", Tier.AUDITED, _discriminates, skip=audit_skip)
    return report
