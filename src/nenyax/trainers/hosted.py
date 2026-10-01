"""Trainers that run in-process: the built-in learners, and Tinker's hosted training service.

These don't launch a separate trainer: ``run`` drives :func:`nenyax.train` with the matching
learner, so the gate, held-out evaluation and round events are the built-in ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..environment import Environment
from .base import Events, Job, JobResult, Requirements, Trainer, TrainerUnavailable, _emit


class InProcessTrainer(Trainer):
    """Shared by the in-process trainers: the job is just the learner spec plus the train() budget."""

    learner_use = "incontext"

    def prepare(self, env: Environment, model: str, config: dict[str, Any], workdir: Path) -> Job:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        learner = {"use": config.get("learner", self.learner_use), "model": model}
        learner.update(config.get("learner_args", {}))
        budget = {k: config[k] for k in ("rounds", "group_size", "batch") if k in config}
        plan = workdir / "nenyax_train.json"
        plan.write_text(json.dumps({"learner": learner, "train": budget}, indent=2))
        return Job(
            trainer=self.id,
            workdir=workdir,
            command=["nenyax", "train", "--config", str(plan)],
            files={"plan": plan},
            meta={"learner": learner, "train": budget, "env": env},
        )

    def run(self, job: Job, events: Events = None, *, timeout: float | None = None) -> JobResult:
        ok, why = self.available()
        if not ok:
            raise TrainerUnavailable(f"{self.name} can't run here: {why}")
        from ..config import build
        from ..train import train

        env = job.meta["env"]
        learner = build("learner", dict(job.meta["learner"]))
        rounds: list[dict[str, Any]] = []

        def on_round(r: Any) -> None:
            row = {"round": r.round, "eval_score": r.eval_score, "kept": r.kept}
            rounds.append(row)
            _emit(events, "round", row)

        result = train(env, learner, on_round=on_round, **job.meta["train"])
        summary = {"baseline": result.baseline, "best": result.best}
        _emit(events, "metric", {"trainer": self.id, **summary})
        return JobResult(0, None, [*rounds, summary], json.dumps(summary))


class NenyaxTrainer(InProcessTrainer):
    id = "nenyax"
    name = "Nenyax learners"
    status = "ready"
    summary = "The built-in learners: in-context (any chat model, no GPU), GRPO, DAPO, RLOO, RFT, DPO and more."
    config_schema = {
        "type": "object",
        "properties": {
            "learner": {
                "enum": [
                    "incontext",
                    "grpo",
                    "dr_grpo",
                    "dapo",
                    "rloo",
                    "reinforce",
                    "reinforce_pp",
                    "gspo",
                    "rft",
                    "dpo",
                ],
                "default": "incontext",
            },
            "rounds": {"type": "integer", "default": 3},
            "group_size": {"type": "integer", "default": 4},
            "batch": {"type": "integer", "default": 6},
        },
    }

    def requirements(self) -> Requirements:
        return Requirements(supports=["text", "multi-turn", "tools"])

    def prepare(self, env, model, config, workdir):
        job = super().prepare(env, model, config, workdir)
        if job.meta["learner"]["use"] == "incontext":  # the in-context learner takes a model spec
            job.meta["learner"]["model"] = config.get("model_spec") or {"name": model}
        return job


class TinkerTrainer(InProcessTrainer):
    id = "tinker"
    name = "Tinker (Thinking Machines)"
    summary = "Hosted LoRA training: Nenyax samples and scores, Tinker updates the weights. Needs TINKER_API_KEY."
    learner_use = "tinker"
    config_schema = {
        "type": "object",
        "properties": {
            "rounds": {"type": "integer", "default": 10},
            "group_size": {"type": "integer", "default": 8},
            "batch": {"type": "integer", "default": 8},
            "learner_args": {
                "type": "object",
                "description": "TinkerLearner options: lr, lora_rank, loss_fn, ...",
            },
        },
    }

    def requirements(self) -> Requirements:
        return Requirements(
            packages=["tinker", "transformers"],
            install=["tinker", "transformers"],
            supports=["text", "multi-turn"],
            env_vars=["TINKER_API_KEY"],
        )
