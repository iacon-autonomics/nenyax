"""The trainer interface: plug the trainers people already use into Nenyax.

A :class:`Trainer` turns a Nenyax environment plus a model into that trainer's native job:

* ``prepare(env, model, config, workdir)`` writes the trainer's config, a dataset of the
  environment's prompts, and a bridge module so rewards come from the environment's verifier;
  it returns a :class:`Job` with the exact launch command. This always works, without the
  trainer installed, so jobs can be prepared here and launched on a GPU elsewhere.
* ``run(job, events)`` launches it, streams its log, turns ``NENYAX_METRIC {...}`` lines into
  round events, and returns where the checkpoint landed.

Every trainer is also a learner (``[learner] use = "trl"``), so ``nenyax.train`` and the
platform pipeline measure the held-out score before and after, like the built-in learners.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ..environment import Environment, NenyaxError

METRIC_PREFIX = "NENYAX_METRIC "
Events = Callable[[str, dict[str, Any]], None] | Any | None


class TrainerUnavailable(NenyaxError):
    """The trainer can't run here: a package or a GPU is missing (prepare() still works)."""


@dataclass
class Requirements:
    gpus: int = 0
    min_vram_gb: int = 0
    packages: list[str] = field(default_factory=list)
    #: pip specs that provide ``packages``.
    install: list[str] = field(default_factory=list)
    #: A container image that has everything, for compute plugs that run images.
    image: str | None = None
    #: What episodes it can train on: text (single reply), multi-turn, tools.
    supports: list[str] = field(default_factory=lambda: ["text"])
    #: Environment variables it needs (e.g. TINKER_API_KEY).
    env_vars: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Job:
    trainer: str
    workdir: Path
    command: list[str]
    files: dict[str, Path] = field(default_factory=dict)
    checkpoint_dir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    #: Extra processes that must run alongside (e.g. OpenRLHF's reward server).
    sidecars: list[list[str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shell(self) -> str:
        return " ".join(shlex.quote(c) for c in self.command)


@dataclass
class JobResult:
    returncode: int
    checkpoint: Path | None
    metrics: list[dict[str, Any]]
    log_tail: str


def _emit(events: Events, kind: str, payload: dict[str, Any]) -> None:
    if events is None:
        return
    if hasattr(events, "emit"):
        events.emit(kind, payload)
    else:
        events(kind, payload)


def has_gpu() -> bool:
    if find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            return True
    from shutil import which

    if which("nvidia-smi"):
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    return False


class Trainer:
    """Base class. Subclasses set the class attributes and implement :meth:`prepare`."""

    id: str = ""
    name: str = ""
    #: ready = run() is implemented and exercised for real; beta = prepared and tested, not yet
    #: run end to end against the real trainer; planned = catalog only.
    status: str = "beta"
    summary: str = ""
    config_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def requirements(self) -> Requirements:
        return Requirements()

    def prepare(self, env: Environment, model: str, config: dict[str, Any], workdir: Path) -> Job:
        raise NotImplementedError

    def available(self) -> tuple[bool, str]:
        """Whether run() can work on this machine, and why not."""
        req = self.requirements()
        missing = [p for p in req.packages if find_spec(p) is None]
        problems = []
        if missing:
            problems.append(
                f"needs {', '.join(missing)} (pip install {' '.join(req.install or missing)})"
            )
        if req.gpus and not has_gpu():
            problems.append(f"needs {req.gpus} CUDA GPU{'s' if req.gpus > 1 else ''}")
        unset = [v for v in req.env_vars if not os.environ.get(v)]
        if unset:
            problems.append(f"needs {', '.join(unset)} set")
        return (not problems, "; ".join(problems))

    def run(self, job: Job, events: Events = None, *, timeout: float | None = None) -> JobResult:
        ok, why = self.available()
        if not ok:
            raise TrainerUnavailable(
                f"{self.name} can't run here: {why}. The job is prepared in {job.workdir}; "
                f"launch it on a GPU machine with: {job.shell}"
            )
        sidecars = [
            subprocess.Popen(c, cwd=job.workdir, env={**os.environ, **job.env})
            for c in job.sidecars
        ]
        metrics: list[dict[str, Any]] = []
        tail: list[str] = []
        try:
            proc = subprocess.Popen(
                job.command,
                cwd=job.workdir,
                env={**os.environ, **job.env},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                tail = (tail + [line])[-200:]
                if line.startswith(METRIC_PREFIX):
                    try:
                        m = json.loads(line[len(METRIC_PREFIX) :])
                    except ValueError:
                        continue
                    metrics.append(m)
                    _emit(events, "metric", {"trainer": self.id, **m})
                else:
                    _emit(events, "log", {"level": "info", "message": line[:2000]})
            proc.wait(timeout=timeout)
        finally:
            for s in sidecars:
                s.terminate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.name} exited with {proc.returncode}:\n" + "\n".join(tail[-40:])
            )
        return JobResult(proc.returncode, job.checkpoint_dir, metrics, "\n".join(tail[-40:]))

    def artifacts(self, job: Job) -> list[dict[str, Any]]:
        """What a finished job leaves behind, for the platform's storage plug."""
        out = []
        if job.checkpoint_dir and job.checkpoint_dir.exists():
            out.append(
                {
                    "kind": "checkpoint",
                    "name": f"{self.id} checkpoint",
                    "path": str(job.checkpoint_dir),
                }
            )
        for name, path in job.files.items():
            if path.exists() and name in ("train_data", "eval_data"):
                out.append({"kind": "dataset", "name": name, "path": str(path)})
        return out

    def catalog_entry(self) -> dict[str, Any]:
        req = self.requirements()
        needs = [f"{req.gpus} GPU{'s' if req.gpus != 1 else ''}"] if req.gpus else []
        needs += req.env_vars
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "needs": needs,
            "requirements": req.public(),
            "config_schema": self.config_schema,
        }


def python(config: dict[str, Any]) -> str:
    """The interpreter that runs the trainer (its own venv if it has one)."""
    return str(config.get("python") or sys.executable)
