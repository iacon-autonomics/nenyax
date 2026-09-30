"""Prime Intellect Verifiers driver, in rollout mode. Supports both Verifiers stacks.

* **Legacy** (``import verifiers as vf``): ``vf.Environment`` objects such as ``SingleTurnEnv``,
  ``MultiTurnEnv`` and ``ToolEnv``, loaded with ``vf.load_environment(id)``.
* **v1** (``verifiers.v1``): Taskset/Harness/Runtime environments. Environments Hub packages
  (``vf-install owner/name``) are v1 tasksets.

The environment runs the whole episode itself, calling the model endpoint Nenyax provides, so every
model call is recorded by the Nenyax model server::

    nenyax.load("verifiers:gsm8k")                        # v1 taskset (hub package)
    nenyax.load("verifiers:my_env?args={\\"n\\": 50}")    # legacy load_environment(**args)
    VerifiersEnvironment(my_vf_env)                       # an Environment object you built
"""

from __future__ import annotations

import asyncio
import importlib
import os
from typing import Any

from .._util import run_sync, to_json
from ..environment import ModelDrivenEnvironment
from ..policy import Endpoint, FunctionModel
from ..registry import Driver
from ..types import Capabilities, Judgment, Manifest, Message, Mode, TaskRef

_KEY_VAR = "NENYAX_VERIFIERS_API_KEY"


def _message(m: Any) -> Message:
    d = m if isinstance(m, dict) else m.model_dump()
    d = {k: v for k, v in d.items() if v is not None}
    role = d.get("role", "user")
    content = d.get("content")
    if content is not None and not isinstance(content, (str, list)):
        content = str(content)
    return Message(
        role=role if role in ("system", "user", "assistant", "tool") else "user",
        content=content,
        tool_call_id=d.get("tool_call_id"),
    )


def _messages(value: Any) -> list[Message]:
    if value is None:
        return []
    if isinstance(value, str):
        return [Message(role="user", content=value)]
    return [_message(m) for m in value]


class VerifiersEnvironment(ModelDrivenEnvironment):
    def __init__(
        self,
        env: Any = None,
        *,
        env_id: str | None = None,
        args: dict[str, Any] | None = None,
        stack: str = "auto",
        harness: str = "null",
        runtime: str = "subprocess",
        sampling: dict[str, Any] | None = None,
        reference: str | None = None,
        null: str = "",
    ) -> None:
        """
        Args:
            env: an already-built legacy ``vf.Environment`` (skips loading).
            env_id: module / hub id to load.
            args: kwargs for a legacy ``load_environment``.
            stack: ``"legacy"``, ``"v1"``, or ``"auto"`` (legacy if the module defines
                ``load_environment``, else v1).
            harness / runtime: v1 agent configuration. ``null`` + ``subprocess`` is a plain chat
                loop on the local machine (the v1 defaults need Prime Intellect auth).
            sampling: sampling args forwarded to every model call.
            reference: format string over a task's fields yielding a correct reply, e.g.
                ``"<answer>{answer}</answer>"``; enables the audit tier.
            null: reply expected to score zero, used by audits.
        """
        import verifiers

        self.sampling = dict(sampling or {"temperature": 0.0, "max_tokens": 512})
        self.reference_template, self.null_reply = reference, null
        self.stack = stack
        if env is not None:
            self.stack = "legacy"
            self.vf_env = env
            env_id = env_id or getattr(env, "env_id", None) or type(env).__name__
        else:
            if not env_id:
                raise ValueError("pass a verifiers Environment or an env_id")
            if stack == "auto":
                module = env_id.replace("-", "_")
                try:
                    has_legacy = hasattr(importlib.import_module(module), "load_environment")
                except ImportError:
                    has_legacy = False
                self.stack = "legacy" if has_legacy else "v1"
            if self.stack == "legacy":
                self.vf_env = verifiers.load_environment(env_id, **(args or {}))
            else:
                self.vf_env = self._load_v1(env_id, harness, runtime)
        self._rows = self._load_rows()
        self.manifest = Manifest(
            id=f"verifiers:{env_id}",
            name=str(env_id),
            source_format=f"verifiers/{self.stack}",
            mode=Mode.ROLLOUT,
            version=getattr(verifiers, "__version__", None),
            capabilities=Capabilities(
                resettable=True,
                text=True,
                reward_timing="terminal",
                requires=["model_endpoint"],
                token_capture="proxy",
            ),
            num_tasks=len(self._rows),
            extra={"stack": self.stack, "harness": harness if self.stack == "v1" else None},
        )

    # -- loading -------------------------------------------------------------------------------

    @staticmethod
    def _load_v1(env_id: str, harness: str, runtime: str) -> Any:
        import verifiers.v1 as vf1
        from verifiers.v1.envs.single_agent import SingleAgentEnvConfig
        from verifiers.v1.utils.loaders import load_environment

        harness_cfg: Any
        if harness == "null":
            from verifiers.v1.harnesses.null import NullHarnessConfig

            harness_cfg = NullHarnessConfig(id="null")
        else:
            harness_cfg = {"id": harness}
        runtime_cfg: Any
        if runtime == "subprocess":
            from verifiers.v1.runtimes import SubprocessConfig

            runtime_cfg = SubprocessConfig()
        else:
            runtime_cfg = {"type": runtime}
        cfg = SingleAgentEnvConfig(
            taskset=vf1.TasksetConfig(id=env_id),
            agent=vf1.AgentConfig(harness=harness_cfg, runtime=runtime_cfg),
        )
        return load_environment(cfg)

    def _load_rows(self) -> list[Any]:
        if self.stack == "legacy":
            return list(self.vf_env._get_eval_inputs(num_examples=-1, rollouts_per_example=1))
        return list(self.vf_env.taskset.head(10_000))

    def _fields(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        if self.stack == "legacy":
            return dict(row)
        data = row.data
        return data.model_dump() if hasattr(data, "model_dump") else dict(vars(data))

    # -- tasks and audit policies --------------------------------------------------------------

    def tasks(self, split: str | None = None, limit: int | None = None) -> list[TaskRef]:
        refs = []
        for i in range(len(self._rows))[:limit]:
            fields = self._fields(i)
            prompt = fields.get("question") or fields.get("prompt")
            refs.append(TaskRef(id=str(i), prompt=prompt if isinstance(prompt, str) else None))
        return refs

    def _index(self, task: TaskRef | None, seed: int | None) -> int:
        return int(task.id) if task else (seed or 0) % len(self._rows)

    def reference_policy(self, task: TaskRef | None = None) -> FunctionModel | None:
        if self.reference_template is None or task is None:
            return None
        reply = self.reference_template.format(**self._fields(int(task.id)))
        return FunctionModel(lambda messages: reply, name="nenyax-reference")

    def null_policy(self, task: TaskRef | None = None) -> FunctionModel:
        reply = self.null_reply
        return FunctionModel(lambda messages: reply, name="nenyax-null")

    # -- running -------------------------------------------------------------------------------

    def _run(
        self, endpoint: Endpoint, *, task: TaskRef | None, seed: int | None, max_steps: int | None
    ) -> tuple[Judgment, list[Message], dict[str, Any]]:
        index = self._index(task, seed)
        if self.stack == "legacy":
            return run_sync(self._run_legacy(endpoint, index))
        return run_sync(self._run_v1(endpoint, index))

    async def _run_legacy(self, endpoint: Endpoint, index: int):
        import verifiers as vf
        from openai import AsyncOpenAI

        client = vf.OpenAIChatCompletionsClient(
            AsyncOpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key)
        )
        out = await self.vf_env.run_rollout(
            dict(self._rows[index]), client, endpoint.model, dict(self.sampling)
        )
        if out.get("error"):
            raise RuntimeError(f"verifiers rollout error: {out['error']}")
        metrics = {k: float(v) for k, v in (out.get("metrics") or {}).items()}
        judgment = Judgment(
            score=float(out["reward"]),
            components=metrics,
            source="rubric",
            judge=f"verifiers:{type(self.vf_env).__name__}.rubric",
            details={"stop_condition": to_json(out.get("stop_condition"))},
        )
        messages = _messages(out.get("prompt")) + _messages(out.get("completion"))
        return judgment, messages, {"example_id": to_json(out.get("example_id"))}

    async def _run_v1(self, endpoint: Endpoint, index: int):
        import verifiers.v1 as vf1

        os.environ[_KEY_VAR] = endpoint.api_key or "nenyax"
        ctx = vf1.ModelContext(
            model=endpoint.model,
            client=vf1.EvalClientConfig(base_url=endpoint.base_url, api_key_var=_KEY_VAR),
            sampling=vf1.SamplingConfig(**self.sampling),
        )
        async with self.vf_env.serving():
            slot = self.vf_env.slots(self._rows[index], 1)[0]
            episode = await self.vf_env.run_slot(slot, ctx, asyncio.Semaphore(1))
        trace = episode.traces[0]
        errors = [e.message for e in list(episode.errors) + list(trace.errors)]
        if errors or trace.reward is None:
            raise RuntimeError(f"verifiers v1 episode failed: {errors or 'no reward'}")
        components = {k: float(r.score) for k, r in trace.rewards.items() if r is not None}
        components.update(
            {k: float(v) for k, v in (trace.metrics or {}).items() if isinstance(v, (int, float))}
        )
        judgment = Judgment(
            score=float(trace.reward),
            components=components,
            source="rubric",
            judge=f"verifiers.v1:{type(self._rows[index]).__name__}",
            details={"stop_condition": to_json(trace.stop_condition)},
        )
        return judgment, _messages(trace.messages), {"episode_id": str(episode.id)}


class VerifiersDriver(Driver):
    name = "verifiers"
    format = "Verifiers (Prime Intellect)"
    module = "verifiers"
    extra = "verifiers"

    def load(self, target: str, **options: Any) -> VerifiersEnvironment:
        self.require("verifiers")
        return VerifiersEnvironment(env_id=target, **options)
