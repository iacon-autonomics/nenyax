"""Trainers: the training frameworks people already use, fed by Nenyax environments.

==========  =========================================================================  ======
nenyax      the built-in learners (in-context, GRPO, DAPO, RLOO, RFT, DPO, ...)         ready
trl         TRL GRPO (reward = the environment's verifier) or DPO                       beta
unsloth     GRPO through TRL with Unsloth's 4-bit loading and LoRA                       beta
verl        verl GRPO/PPO; verl parquet + a custom reward function                       beta
prime-rl    prime-rl on a verifiers environment that wraps the Nenyax environment        beta
openrlhf    OpenRLHF PPO/GRPO with Nenyax as the remote reward model                     beta
tinker      Thinking Machines Tinker, hosted LoRA (TINKER_API_KEY)                       beta
==========  =========================================================================  ======

``beta`` means ``prepare()`` generates the job and is tested, and the rewards are tested against
real environments, but the job has not been run end to end on a GPU against the real trainer.

    trainer = nenyax.trainers.get("trl")
    job = trainer.prepare(env, "Qwen/Qwen2.5-0.5B-Instruct", {"max_steps": 100}, "./job")
    print(job.shell)              # launch it anywhere, or:
    trainer.run(job)              # here, if trl and a GPU are present

Each is also a learner, so ``[learner] use = "trl"`` works in configs and platform pipelines.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .base import Job, JobResult, Requirements, Trainer, TrainerUnavailable

__all__ = [
    "ExternalLearner",
    "Job",
    "JobResult",
    "Requirements",
    "Trainer",
    "TrainerUnavailable",
    "catalog",
    "get",
    "trainers",
]

_CLASSES = {
    "nenyax": "hosted:NenyaxTrainer",
    "trl": "trl:TRLTrainer",
    "unsloth": "trl:UnslothTrainer",
    "verl": "verl:VerlTrainer",
    "prime-rl": "prime_rl:PrimeRLTrainer",
    "openrlhf": "openrlhf:OpenRLHFTrainer",
    "tinker": "hosted:TinkerTrainer",
}
ALIASES = {"prime_rl": "prime-rl"}


def get(name: str) -> Trainer:
    import importlib

    key = ALIASES.get(name, name)
    if key not in _CLASSES:
        raise KeyError(f"unknown trainer {name!r}; one of {sorted(_CLASSES)}")
    module, _, cls = _CLASSES[key].partition(":")
    return getattr(importlib.import_module(f".{module}", __name__), cls)()


def trainers() -> list[str]:
    return list(_CLASSES)


def catalog() -> list[dict[str, Any]]:
    """Every trainer plug, for the platform's ``/plugs`` and ``nenyax trainers``."""
    out = []
    for name in _CLASSES:
        t = get(name)
        entry = t.catalog_entry()
        ok, why = t.available()
        entry["runs_here"], entry["why_not_here"] = ok, (None if ok else why)
        out.append(entry)
    return out


class ExternalLearner:
    """A trainer as a learner, so ``nenyax.train`` (and the platform pipeline) can drive it.

    Each ``update`` runs one training job (resuming from the previous checkpoint); the policy
    before and after comes from ``eval_model``: an endpoint spec serving the base model (e.g. a
    vLLM server). After training, the policy is ``eval_after`` if given, else ``eval_model``
    with the checkpoint as its model name (vLLM serving the checkpoint directory).
    """

    def __init__(
        self,
        trainer: str | Trainer,
        model: str,
        *,
        eval_model: dict[str, Any] | None = None,
        eval_after: dict[str, Any] | None = None,
        workdir: str | None = None,
        **config: Any,
    ) -> None:
        self.trainer = get(trainer) if isinstance(trainer, str) else trainer
        self.model, self.config = model, config
        self.eval_model, self.eval_after = eval_model, eval_after
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix=f"nenyax-{self.trainer.id}-"))
        self.env = None
        self.checkpoint: str | None = None
        self.jobs: list[Job] = []

    def bind(self, env: Any, tasks: Any = None, eval_tasks: Any = None) -> None:
        """Called by nenyax.train before the baseline: the trainer needs the environment."""
        self.env = env

    def policy(self) -> Any:
        from ..config import build

        if self.checkpoint and self.eval_after is not None:
            return build("model", self.eval_after)
        if self.eval_model is None:
            raise ValueError(
                f"{self.trainer.name} trains outside Nenyax; to measure before and after, pass "
                'eval_model={"provider": "vllm", "name": "<model>", "base_url": "..."} '
                "(an endpoint serving the model)"
            )
        if not isinstance(self.eval_model, dict):  # an already-built policy
            return self.eval_model
        spec = dict(self.eval_model)
        if self.checkpoint:
            spec["name"] = self.checkpoint
        return build("model", spec)

    def update(self, groups: Any) -> dict[str, float]:
        if self.env is None:
            raise RuntimeError("bind(env) first (nenyax.train does this)")
        model = self.checkpoint or self.model
        job = self.trainer.prepare(
            self.env, model, self.config, self.workdir / f"round{len(self.jobs) + 1}"
        )
        self.jobs.append(job)
        result = self.trainer.run(job)
        if result.checkpoint:
            self.checkpoint = str(result.checkpoint)
        numeric = [m for m in result.metrics if isinstance(m, dict)]
        last = numeric[-1] if numeric else {}
        return {k: float(v) for k, v in last.items() if isinstance(v, (int, float))}

    def snapshot(self) -> Any:
        return self.checkpoint

    def restore(self, state: Any) -> None:
        self.checkpoint = state
