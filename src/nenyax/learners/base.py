"""The learner contract: anything that turns judged experience into a better policy.

    class MyLearner:
        def policy(self) -> Policy: ...                         # act with the current version
        def update(self, groups: list[list[Trajectory]]) -> dict[str, float]: ...

Optional: ``snapshot() -> Any`` and ``restore(snapshot)``, so :func:`nenyax.train` can keep only
updates that improve held-out performance.

``groups`` are rollouts of the same task, which is what group-relative methods (GRPO, RLOO) and
best-of-n selection need. Learners register through the ``nenyax.learners`` entry-point group.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from ..environment import Policy
from ..types import Trajectory


@runtime_checkable
class Learner(Protocol):
    def policy(self) -> Policy: ...

    def update(self, groups: list[list[Trajectory]]) -> dict[str, float]: ...


def group_advantages(
    groups: list[list[Trajectory]], *, normalize: bool = True, eps: float = 1e-6
) -> list[list[float]]:
    """Group-relative advantages: each score minus its group's mean (optionally / std).

    Groups where every episode scored the same carry no signal and get all-zero advantages.
    """
    out = []
    for group in groups:
        scores = [t.score or 0.0 for t in group]
        mean = sum(scores) / len(scores) if scores else 0.0
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores)) if scores else 0.0
        if std < eps:
            out.append([0.0] * len(scores))
        else:
            out.append([(s - mean) / (std if normalize else 1.0) for s in scores])
    return out


def has_signal(groups: list[list[Trajectory]]) -> float:
    """Fraction of groups whose scores differ (the rest teach nothing)."""
    if not groups:
        return 0.0
    live = sum(1 for g in groups if len({t.score for t in g}) > 1)
    return live / len(groups)


def snapshot(learner: Any) -> Any:
    fn = getattr(learner, "snapshot", None)
    return fn() if callable(fn) else None


def restore(learner: Any, state: Any) -> bool:
    fn = getattr(learner, "restore", None)
    if callable(fn) and state is not None:
        fn(state)
        return True
    return False
