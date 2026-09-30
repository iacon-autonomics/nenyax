"""Learners: turn judged experience into a better policy.

Built in, all selectable by name in configs (``[learner] use = "..."``):

============== ===============================================================================
incontext      worked examples from the model's own best episodes; any chat model, closed APIs too
grpo           GRPO (group mean/std advantages, clipped ratio)
dr_grpo        Dr. GRPO (no std normalisation, token-level loss)
dapo           DAPO (clip-higher, token-level loss, zero-signal groups dropped)
rloo           RLOO (leave-one-out baseline)
reinforce      REINFORCE with a batch-mean baseline
reinforce_pp   REINFORCE++-style batch-normalised advantages
gspo           GSPO (sequence-level importance ratio)
pg             the policy-gradient learner with every knob exposed (advantage, loss, clip, ...)
rft            rejection-sampling fine-tuning (STaR / expert iteration)
dpo            DPO on best-vs-worst pairs from each group
tinker         Thinking Machines Tinker, hosted LoRA (bring your own key)
============== ===============================================================================

The weight-updating learners need ``pip install 'nenyax[train]'`` (torch, transformers).
Third-party learners register through the ``nenyax.learners`` entry-point group.
"""

from .advantages import ESTIMATORS
from .base import Learner, group_advantages, has_signal
from .incontext import InContextLearner

__all__ = ["ESTIMATORS", "InContextLearner", "Learner", "group_advantages", "has_signal"]

_LAZY = {  # these import torch or vendor SDKs, so only on demand
    "GRPOLearner": "grpo",
    "PolicyGradientLearner": "policy_gradient",
    "RejectionSamplingLearner": "policy_gradient",
    "DPOLearner": "policy_gradient",
    "PRESETS": "policy_gradient",
    "TinkerLearner": "tinker",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
