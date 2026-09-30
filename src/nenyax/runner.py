"""Batch rollouts: the loop every trainer and evaluator needs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TextIO

from .environment import Environment, Policy
from .types import TaskRef, Trajectory


def run(
    env: Environment,
    policy: Policy | Callable[[], Policy],
    *,
    tasks: Iterable[TaskRef | None] | None = None,
    n: int | None = None,
    seed: int = 0,
    concurrency: int = 1,
    max_steps: int | None = None,
    out: str | Path | TextIO | None = None,
    callbacks: Iterable[object] = (),
) -> Iterator[Trajectory]:
    """Run episodes and yield trajectories in order.

    ``policy`` may be a policy or a zero-argument factory; pass a factory when the policy is
    stateful and ``concurrency > 1``. ``out`` streams trajectories as JSON Lines. Each callback's
    ``on_trajectory(traj)`` is called as episodes finish (see ``nenyax.integrations.telemetry``).
    """
    callbacks = list(callbacks)
    if tasks is None:
        listed = env.tasks(limit=n)
        tasks = listed if listed else [None] * (n or 1)
    jobs = [(t, seed + i) for i, t in enumerate(tasks)][: n or None]
    factory = policy if _is_factory(policy) else (lambda: policy)

    def one(job: tuple[TaskRef | None, int]) -> Trajectory:
        task, s = job
        return env.rollout(factory(), task=task, seed=s, max_steps=max_steps)  # type: ignore[operator]

    sink, close = _open_sink(out)
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            for traj in pool.map(one, jobs):
                if sink:
                    sink.write(traj.model_dump_json() + "\n")
                    sink.flush()
                for callback in callbacks:
                    callback.on_trajectory(traj)  # type: ignore[attr-defined]
                yield traj
    finally:
        if close:
            sink.close()  # type: ignore[union-attr]


def _is_factory(policy: object) -> bool:
    """Heuristic: a zero-argument callable that is not itself a policy."""
    import inspect

    if hasattr(policy, "complete") or not callable(policy):
        return False
    try:
        sig = inspect.signature(policy)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    required = [
        p
        for p in sig.parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return not required


def _open_sink(out: str | Path | TextIO | None) -> tuple[TextIO | None, bool]:
    if out is None:
        return None, False
    if isinstance(out, (str, Path)):
        return open(out, "a", encoding="utf-8"), True
    return out, False
