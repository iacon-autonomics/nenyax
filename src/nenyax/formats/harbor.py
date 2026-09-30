"""Harbor driver: containerized agent tasks (Terminal-Bench and 80+ adapted benchmarks), task mode.

A Harbor task is a directory with ``task.toml``, ``instruction.md``, ``environment/Dockerfile``
and ``tests/test.sh``; the tests write the reward after the agent finishes. Nenyax runs one
Harbor *trial* per rollout:

* a :class:`~nenyax.policy.ChatModel` or :class:`~nenyax.policy.Endpoint` is plugged in behind
  a Harbor agent (``terminus-2`` by default) through the recording model server;
* a :class:`~nenyax.policy.NativeAgent` runs any Harbor agent by name (``oracle``, ``nop``,
  ``claude-code``, ...). ``oracle`` and ``nop`` double as the reference and null policies.

URIs::

    harbor:./path/to/task            # one task
    harbor:./path/to/dataset         # every task directory underneath
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

from .._util import run_sync, to_json
from ..environment import ModelDrivenEnvironment
from ..policy import Endpoint, FunctionModel, NativeAgent
from ..registry import Driver
from ..types import Capabilities, Isolation, Judgment, Manifest, Message, Mode, TaskRef

#: litellm needs model metadata for unknown (self-hosted) model names.
_MODEL_INFO = {
    "max_input_tokens": 128_000,
    "max_output_tokens": 8_192,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
}


def _is_task(path: Path) -> bool:
    return (path / "task.toml").is_file()


def _find_tasks(root: Path) -> list[Path]:
    if _is_task(root):
        return [root]
    return sorted(p.parent for p in root.rglob("task.toml"))


def _judgment(result: Any) -> Judgment:
    verifier = result.verifier_result
    rewards = dict(verifier.rewards or {}) if verifier else {}
    if "reward" in rewards:
        score = float(rewards["reward"])
    elif rewards:
        score = sum(float(v) for v in rewards.values()) / len(rewards)
    else:
        raise RuntimeError("Harbor verifier produced no reward")
    return Judgment(
        score=score,
        passed=score >= 1.0,
        components={k: float(v) for k, v in rewards.items()},
        source="tests",
        judge="harbor:tests/test.sh",
    )


def _messages_from_atif(trial_dir: Path) -> list[Message]:
    """Convert Harbor's ATIF trajectory (agent/trajectory.json) into chat messages."""
    path = trial_dir / "agent" / "trajectory.json"
    if not path.is_file():
        return []
    doc = json.loads(path.read_text())
    out: list[Message] = []
    for step in doc.get("steps", []):
        role = "assistant" if step.get("source") == "agent" else "user"
        text = step.get("message")
        if not isinstance(text, str):
            text = json.dumps(text) if text is not None else None
        out.append(Message(role=role, content=text))
        results = (step.get("observation") or {}).get("results") or []
        if results:
            out.append(Message(role="user", content=json.dumps(to_json(results))[:20_000]))
    return out


class HarborEnvironment(ModelDrivenEnvironment):
    def __init__(
        self,
        path: str,
        *,
        agent: str = "terminus-2",
        agent_kwargs: dict[str, Any] | None = None,
        trials_dir: str | None = None,
        environment: str = "docker",
        keep_trials: bool = True,
    ) -> None:
        """
        Args:
            path: a task directory, or a directory containing task directories.
            agent: Harbor agent that wraps a ChatModel/Endpoint policy.
            agent_kwargs: extra kwargs for that agent.
            trials_dir: where Harbor writes trial artifacts (default ``./.nenyax/harbor-trials``).
            environment: Harbor environment type (``docker``, ``daytona``, ``modal``, ...).
            keep_trials: keep Harbor's per-trial artifacts (logs, recordings) on disk.
        """
        import harbor

        self.root = Path(path).expanduser().resolve()
        self.task_dirs = _find_tasks(self.root)
        if not self.task_dirs:
            raise FileNotFoundError(f"no Harbor task (task.toml) under {self.root}")
        self.agent = agent
        self.agent_kwargs = dict(agent_kwargs or {})
        self.trials_dir = Path(trials_dir or ".nenyax/harbor-trials").resolve()
        self.environment = environment
        self.keep_trials = keep_trials
        meta = tomllib.loads((self.task_dirs[0] / "task.toml").read_text()).get("metadata", {})
        single = len(self.task_dirs) == 1
        self.manifest = Manifest(
            id=f"harbor:{self.root}",
            name=self.root.name,
            source_format="harbor",
            mode=Mode.TASK,
            version=getattr(harbor, "__version__", None),
            description=meta.get("description") if single else f"{len(self.task_dirs)} tasks",
            capabilities=Capabilities(
                resettable=True,
                branchable=False,
                text=True,
                reward_timing="terminal",
                requires=["docker", "model_endpoint"],
                isolation=Isolation.CONTAINER,
                token_capture="proxy",
            ),
            num_tasks=len(self.task_dirs),
            extra={"agent": agent, "environment": environment},
        )

    # -- tasks ---------------------------------------------------------------------------------

    def tasks(self, split: str | None = None, limit: int | None = None) -> list[TaskRef]:
        refs = []
        for d in self.task_dirs[:limit]:
            instruction = d / "instruction.md"
            refs.append(
                TaskRef(
                    id=str(d.relative_to(self.root)) if d != self.root else d.name,
                    prompt=instruction.read_text() if instruction.is_file() else None,
                    metadata={"path": str(d), "has_solution": (d / "solution").is_dir()},
                )
            )
        return refs

    def _task_dir(self, task: TaskRef | None) -> Path:
        if task is None:
            return self.task_dirs[0]
        path = task.metadata.get("path")
        return Path(path) if path else (self.root / task.id)

    def reference_policy(self, task: TaskRef | None = None) -> NativeAgent | None:
        has_solution = all((d / "solution").is_dir() for d in self.task_dirs)
        return NativeAgent("oracle") if has_solution else None

    def null_policy(self, task: TaskRef | None = None) -> NativeAgent:
        return NativeAgent("nop")

    def probe_policy(self) -> FunctionModel | None:
        if not self.agent.startswith("terminus"):
            return None

        def finish(messages: list[Message]) -> str:  # a valid Terminus reply that ends the task
            return json.dumps(
                {"analysis": "probe", "plan": "stop", "commands": [], "task_complete": True}
            )

        return FunctionModel(finish, name="nenyax-probe")

    # -- running -------------------------------------------------------------------------------

    def _trial(self, task_dir: Path, agent_config: Any) -> Any:
        from harbor.models.trial.config import EnvironmentConfig, TaskConfig, TrialConfig
        from harbor.trial.trial import Trial

        config = TrialConfig(
            task=TaskConfig(path=task_dir),
            trials_dir=self.trials_dir,
            agent=agent_config,
            environment=EnvironmentConfig(type=self.environment, delete=True),
        )

        async def go() -> Any:
            trial = await Trial.create(config)
            return await trial.run()

        return run_sync(go())

    def _finish(self, result: Any) -> tuple[Judgment, list[Message], dict[str, Any]]:
        trial_dir = Path(str(result.trial_uri).removeprefix("file://"))
        meta: dict[str, Any] = {"trial_dir": str(trial_dir), "task_name": result.task_name}
        for phase in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
            timing = getattr(result, phase, None)
            if timing and timing.started_at and timing.finished_at:
                meta[f"{phase}_s"] = (timing.finished_at - timing.started_at).total_seconds()
        if result.agent_result:
            meta["agent_usage"] = to_json(
                result.agent_result.model_dump(
                    include={"n_input_tokens", "n_output_tokens", "cost_usd"}
                )
            )
        if result.exception_info and not result.verifier_result:
            info = result.exception_info
            raise RuntimeError(f"{info.exception_type}: {info.exception_message}")
        judgment, messages = _judgment(result), _messages_from_atif(trial_dir)
        if not self.keep_trials:
            shutil.rmtree(trial_dir, ignore_errors=True)
            meta.pop("trial_dir")
        return judgment, messages, meta

    def _run_native(self, agent: NativeAgent, *, task, seed, max_steps):
        from harbor.models.trial.config import AgentConfig

        cfg = AgentConfig(name=agent.name, **agent.config)
        return self._finish(self._trial(self._task_dir(task), cfg))

    def _run(self, endpoint: Endpoint, *, task, seed, max_steps):
        from harbor.models.trial.config import AgentConfig

        # litellm insists on an OpenAI key even for self-hosted endpoints.
        os.environ.setdefault("OPENAI_API_KEY", endpoint.api_key or "nenyax")
        kwargs = {"api_base": endpoint.base_url, "model_info": _MODEL_INFO, **self.agent_kwargs}
        cfg = AgentConfig(name=self.agent, model_name=f"openai/{endpoint.model}", kwargs=kwargs)
        return self._finish(self._trial(self._task_dir(task), cfg))


class HarborDriver(Driver):
    name = "harbor"
    format = "Harbor (Laude Institute)"
    module = "harbor"
    extra = "harbor"

    def detect(self, path: Path) -> bool:
        return path.is_dir() and bool(_find_tasks(path))

    def load(self, target: str, **options: Any) -> HarborEnvironment:
        self.require("harbor")
        return HarborEnvironment(target, **options)
