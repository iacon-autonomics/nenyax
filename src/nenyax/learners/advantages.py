"""Advantage estimators for verifiable-reward RL. Pure functions over groups of scores.

Every estimator maps ``groups`` (rollouts of the same task) to per-rollout advantages. A group
whose scores are all equal carries no signal and gets zeros under every estimator.

========== ====================================================================================
grpo       (r - mean) / std within the group                     (DeepSeekMath GRPO)
dr_grpo    r - mean, no std division                              (Dr. GRPO: removes the bias)
rloo       r - mean of the *other* rollouts in the group          (RLOO, leave-one-out)
reinforce  r - mean over the whole batch                          (REINFORCE with a baseline)
batch_norm (r - group mean) / std over the whole batch            (REINFORCE++-style)
========== ====================================================================================
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ..types import Trajectory

Estimator = Callable[[list[list[float]]], list[list[float]]]
EPS = 1e-6


def _std(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _flat(groups: list[list[float]]) -> list[float]:
    return [s for g in groups for s in g]


def grpo(groups: list[list[float]]) -> list[list[float]]:
    out = []
    for g in groups:
        mean, std = sum(g) / len(g), _std(g)
        out.append([0.0] * len(g) if std < EPS else [(s - mean) / std for s in g])
    return out


def dr_grpo(groups: list[list[float]]) -> list[list[float]]:
    out = []
    for g in groups:
        mean = sum(g) / len(g)
        out.append([0.0] * len(g) if _std(g) < EPS else [s - mean for s in g])
    return out


def rloo(groups: list[list[float]]) -> list[list[float]]:
    out = []
    for g in groups:
        n = len(g)
        if n < 2 or _std(g) < EPS:
            out.append([0.0] * n)
            continue
        total = sum(g)
        out.append([s - (total - s) / (n - 1) for s in g])
    return out


def reinforce(groups: list[list[float]]) -> list[list[float]]:
    flat = _flat(groups)
    if not flat:
        return []
    mean = sum(flat) / len(flat)
    return [[0.0] * len(g) if _std(g) < EPS else [s - mean for s in g] for g in groups]


def batch_norm(groups: list[list[float]]) -> list[list[float]]:
    centered = dr_grpo(groups)
    flat = [a for g in centered for a in g if a != 0.0]
    std = _std(flat) if len(flat) > 1 else 0.0
    if std < EPS:
        return centered
    return [[a / std for a in g] for g in centered]


ESTIMATORS: dict[str, Estimator] = {
    "grpo": grpo,
    "dr_grpo": dr_grpo,
    "rloo": rloo,
    "reinforce": reinforce,
    "batch_norm": batch_norm,
}


def compute(name: str, groups: list[list[Trajectory]]) -> list[list[float]]:
    try:
        estimator = ESTIMATORS[name]
    except KeyError:
        known = sorted(ESTIMATORS)
        raise ValueError(f"unknown advantage estimator {name!r}; have {known}") from None
    return estimator([[t.score or 0.0 for t in g] for g in groups])
