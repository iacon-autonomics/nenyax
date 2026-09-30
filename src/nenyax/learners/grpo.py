"""GRPO: the ``grpo`` preset of :class:`~nenyax.learners.policy_gradient.PolicyGradientLearner`.

Kept as its own name because it is the most common entry point::

    learner = GRPOLearner("Qwen/Qwen2.5-0.5B-Instruct", lr=1e-6)

See ``policy_gradient.PRESETS`` for DAPO, Dr. GRPO, RLOO, GSPO, REINFORCE++ and friends.
"""

from __future__ import annotations

from typing import Any

from .policy_gradient import PRESETS, PolicyGradientLearner


class GRPOLearner(PolicyGradientLearner):
    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **{**PRESETS["grpo"], **options})

    _samples = staticmethod(PolicyGradientLearner.samples)  # backwards-compatible name
