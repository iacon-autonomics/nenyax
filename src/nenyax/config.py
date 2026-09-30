"""Configuration: pick every piece by name, never by importing a vendor class.

Every component is a *spec*: a registered name, a ``module:Attr`` path, or a table with ``use``
plus keyword arguments. The same spec works in Python, TOML, YAML or JSON::

    # run.toml
    [env]
    uri = "reasoning_gym:basic_arithmetic?size=200"

    [model]                       # the policy for rollouts (any provider in `nenyax integrations`)
    provider = "vllm"
    name = "Qwen/Qwen3-8B"
    temperature = 1.0

    [judge]                       # optional: re-score the environment
    use = "weighted"
    parts = [{use = "env", weight = 1.0}, {use = "regex", pattern = "<answer>", weight = 0.1}]

    [learner]                     # incontext | grpo | tinker | my_pkg.learners:MyLearner
    use = "grpo"
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    lr = 1e-6

    [train]
    rounds = 10
    group_size = 8
    batch = 8

    [[telemetry]]
    use = "mlflow"
    experiment = "arith"

Run it with ``nenyax train --config run.toml`` or ``nenyax.config.run("run.toml")``.
Third-party components register through entry points (``nenyax.learners``, ``nenyax.judges``,
``nenyax.telemetry``) or are referenced directly as ``package.module:Class``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from .environment import Environment, NenyaxError

Spec = str | dict[str, Any]


class ConfigError(NenyaxError):
    pass


def _import(path: str) -> Any:
    module, _, attr = path.partition(":")
    if not attr:
        raise ConfigError(f"{path!r} is neither a registered name nor a 'module:Attr' path")
    return getattr(importlib.import_module(module), attr)


def _lazy(path: str) -> Callable[..., Any]:
    return lambda **kw: _import(path)(**kw)


def _model(**kw: Any) -> Any:
    from .integrations import endpoint

    provider, name = kw.pop("provider", "vllm"), kw.pop("name", None) or kw.pop("model", None)
    if not name:
        raise ConfigError("a model spec needs name = '<model id>'")
    return endpoint(
        provider, name, base_url=kw.pop("base_url", None), api_key=kw.pop("api_key", None), **kw
    )


def _incontext(**kw: Any) -> Any:
    from .learners import InContextLearner

    return InContextLearner(build("model", kw.pop("model")), **kw)


def _weighted(parts: list[dict[str, Any]]) -> Any:
    from .judges import Weighted

    pieces = {}
    for part in parts:
        part = dict(part)
        weight = float(part.pop("weight", 1.0))
        pieces[build("judge", part)] = weight
    return Weighted(pieces)


def _llm_judge(**kw: Any) -> Any:
    from .judges import LLMJudge

    return LLMJudge(build("model", kw.pop("model")), **kw)


def _preset(name: str) -> Callable[..., Any]:
    def make(**kw: Any) -> Any:
        cls = _import("nenyax.learners.policy_gradient:PolicyGradientLearner")
        clip = kw.pop("clip", None)
        if clip is not None:
            kw["clip"] = tuple(clip)
        return cls.preset(name, kw.pop("model"), **kw)

    return make


BUILTIN: dict[str, dict[str, Callable[..., Any]]] = {
    "learner": {
        "incontext": _incontext,
        **{
            name: _preset(name)
            for name in ("grpo", "dr_grpo", "dapo", "rloo", "reinforce", "reinforce_pp", "gspo")
        },
        "pg": lambda **kw: _import("nenyax.learners.policy_gradient:PolicyGradientLearner")(
            kw.pop("model"), **kw
        ),
        "rft": lambda **kw: _import("nenyax.learners.policy_gradient:RejectionSamplingLearner")(
            kw.pop("model"), **kw
        ),
        "dpo": lambda **kw: _import("nenyax.learners.policy_gradient:DPOLearner")(
            kw.pop("model"), **kw
        ),
        "tinker": lambda **kw: _import("nenyax.learners.tinker:TinkerLearner")(
            kw.pop("model"), **kw
        ),
    },
    "judge": {
        "env": _lazy("nenyax.judges:EnvJudge"),
        "exact_match": _lazy("nenyax.judges:ExactMatch"),
        "regex": _lazy("nenyax.judges:Regex"),
        "llm": _llm_judge,
        "weighted": _weighted,
    },
    "telemetry": {
        "opentelemetry": _lazy("nenyax.integrations.telemetry:OpenTelemetry"),
        "mlflow": _lazy("nenyax.integrations.telemetry:MLflow"),
        "wandb": _lazy("nenyax.integrations.telemetry:WandB"),
    },
    "sandbox": {},  # filled from nenyax.sandbox's own registry
    "model": {"endpoint": _model},
}

_GROUPS = {"learner": "nenyax.learners", "judge": "nenyax.judges", "telemetry": "nenyax.telemetry"}


def registered(kind: str) -> dict[str, Callable[..., Any]]:
    """Every name usable for ``kind``: built-ins, then entry-point plugins."""
    found = dict(BUILTIN.get(kind, {}))
    if kind == "sandbox":
        from . import sandbox

        found.update({n: (lambda n=n, **kw: sandbox.get(n, **kw)) for n in sandbox.backends()})
    group = _GROUPS.get(kind)
    if group:
        for ep in entry_points(group=group):
            found[ep.name] = lambda ep=ep, **kw: ep.load()(**kw)
    return found


def build(kind: str, spec: Spec | Any) -> Any:
    """Turn a spec into a live object. Already-built objects pass through unchanged."""
    if not isinstance(spec, (str, dict)):
        return spec
    if kind == "model" and isinstance(spec, dict) and "use" not in spec:
        return _model(**spec)
    if kind == "env":
        return build_env(spec)
    options: dict[str, Any] = {}
    if isinstance(spec, dict):
        options = dict(spec)
        name = options.pop("use", None)
        if name is None:
            raise ConfigError(f"{kind} spec needs 'use': {spec}")
    else:
        name = spec
    factories = registered(kind)
    if name in factories:
        return factories[name](**options)
    if ":" in name:
        return _import(name)(**options)
    raise ConfigError(f"unknown {kind} {name!r}; registered: {sorted(factories)}")


def build_env(spec: Spec) -> Environment:
    from .registry import load

    if isinstance(spec, str):
        return load(spec)
    options = dict(spec)
    judge = options.pop("judge", None)
    uri, factory = options.pop("uri", None), options.pop("use", None)
    if uri:
        env = load(uri, **options)
    elif factory:  # any callable returning an Environment, e.g. "my_envs:make_claims_env"
        env = _import(factory)(**options)
    else:
        raise ConfigError("an env spec needs uri = '<driver>:<target>' or use = 'module:factory'")
    if judge is not None:
        from .judges import with_judge

        env = with_judge(env, build("judge", judge))
    return env


def read(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".toml":
        import tomllib

        return tomllib.loads(text)
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ConfigError("pip install pyyaml to read YAML configs") from e
        return yaml.safe_load(text)
    return json.loads(text)


def run(config: str | Path | dict[str, Any]) -> Any:
    """Build everything a config describes and run ``nenyax.train`` with it."""
    from .train import train

    cfg = read(config) if not isinstance(config, dict) else config
    env = cfg["env"] if isinstance(cfg["env"], Environment) else build_env(cfg["env"])
    if "judge" in cfg:
        from .judges import with_judge

        env = with_judge(env, build("judge", cfg["judge"]))
    learner_spec = cfg.get("learner") or {"use": "incontext"}
    if isinstance(learner_spec, dict):  # an already-built learner object also works
        learner_spec = dict(learner_spec)
        if learner_spec.get("use", "incontext") == "incontext" and "model" not in learner_spec:
            learner_spec["model"] = cfg.get("model") or {}
    learner = build("learner", learner_spec)
    callbacks = [build("telemetry", t) for t in cfg.get("telemetry", [])]
    return train(env, learner, callbacks=callbacks, **cfg.get("train", {}))
