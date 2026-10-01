"""The improvement loop: sample groups, judge, update, keep only what helps.

    result = nenyax.train(env, learner, rounds=5, group_size=4, batch=8)
    print(result.baseline, result.best, result.history)

Every piece is swappable: any environment (any format, any judge), any learner (your algorithm,
a local trainer, a hosted platform), any callbacks. It is a reference loop, not a framework; copy
it when you need something different.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .environment import Environment
from .learners.base import Learner, has_signal, restore, snapshot
from .runner import run
from .types import TaskRef, Trajectory


@dataclass
class Round:
    round: int
    train_score: float
    eval_score: float | None
    kept: bool
    signal: float
    learner: dict[str, float]
    duration_s: float
    errors: int = 0


@dataclass
class TrainResult:
    baseline: float | None
    best: float | None
    history: list[Round] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"baseline eval {self._f(self.baseline)} → best {self._f(self.best)}"]
        for r in self.history:
            mark = "kept" if r.kept else "reverted"
            errs = f"  errors {r.errors}" if r.errors else ""
            lines.append(
                f"  round {r.round:>2}  train {r.train_score:.3f}  eval {self._f(r.eval_score)}"
                f"  signal {r.signal:.2f}  {mark:<8} {r.duration_s:5.1f}s{errs}  {r.learner}"
            )
        return "\n".join(lines)

    @staticmethod
    def _f(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"


def _mean(trajs: Iterable[Trajectory]) -> float:
    scores = [t.score or 0.0 for t in trajs]
    return sum(scores) / len(scores) if scores else 0.0


class TrainingError(RuntimeError):
    """Every episode failed, so there is nothing to learn from (the cause is attached)."""


def _check(trajs: list[Trajectory], stage: str) -> list[Trajectory]:
    ok = [t for t in trajs if t.ok]
    if trajs and not ok:
        raise TrainingError(
            f"all {len(trajs)} {stage} episodes failed; first error: {trajs[0].error}"
        )
    return trajs


def evaluate(
    env: Environment, policy, tasks: list[TaskRef] | None, *, n: int = 8, concurrency: int = 1
) -> float:
    trajs = list(run(env, policy, tasks=tasks, n=None if tasks else n, concurrency=concurrency))
    return _mean(_check(trajs, "evaluation"))


def train(
    env: Environment,
    learner: Learner,
    *,
    rounds: int = 5,
    group_size: int = 4,
    batch: int = 8,
    tasks: list[TaskRef] | None = None,
    eval_tasks: list[TaskRef] | None = None,
    gate: bool = True,
    concurrency: int = 1,
    callbacks: Iterable[object] = (),
    seed: int = 0,
    on_round: Callable[[Round], None] | None = None,
) -> TrainResult:
    """Run ``rounds`` of: sample ``batch`` tasks × ``group_size`` rollouts, update, evaluate.

    With ``gate=True`` and a learner that supports snapshot/restore, an update that lowers the
    held-out score is reverted. ``on_round`` is called after every round (for live progress).
    """
    rng = random.Random(seed)
    pool = tasks if tasks is not None else env.tasks()
    if not pool:
        raise ValueError("train() needs a finite task set (env.tasks() or tasks=)")
    if eval_tasks is None:  # hold out a slice so improvements are measured, not memorized
        shuffled = pool[:]
        rng.shuffle(shuffled)
        cut = max(1, len(shuffled) // 4)
        eval_tasks, pool = shuffled[:cut], shuffled[cut:] or shuffled
    callbacks = list(callbacks)
    bind = getattr(learner, "bind", None)
    if callable(bind):  # learners that train outside Nenyax (nenyax.trainers) need the env
        bind(env, pool, eval_tasks)

    result = TrainResult(
        baseline=evaluate(env, learner.policy(), eval_tasks, concurrency=concurrency), best=None
    )
    result.best = result.baseline
    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        chosen = [rng.choice(pool) for _ in range(batch)]
        jobs = [task for task in chosen for _ in range(group_size)]
        trajs = list(
            run(
                env,
                learner.policy,
                tasks=jobs,
                seed=seed + r * 10_000,
                concurrency=concurrency,
                callbacks=callbacks,
            )
        )
        _check(trajs, "training")
        errors = sum(1 for t in trajs if not t.ok)
        groups = [trajs[i : i + group_size] for i in range(0, len(trajs), group_size)]
        before = snapshot(learner) if gate else None
        metrics = learner.update(groups)
        score = evaluate(env, learner.policy(), eval_tasks, concurrency=concurrency)
        kept = True
        if gate and result.best is not None and score < result.best and restore(learner, before):
            kept = False
        elif result.best is None or score > result.best:
            result.best = score
        result.history.append(
            Round(
                r,
                _mean(trajs),
                score,
                kept,
                has_signal(groups),
                {k: round(v, 4) for k, v in metrics.items()},
                time.perf_counter() - t0,
                errors,
            )
        )
        if on_round is not None:
            on_round(result.history[-1])
    return result
