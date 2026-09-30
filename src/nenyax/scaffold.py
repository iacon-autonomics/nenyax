"""``nenyax new-plugin <kind> <name>``: a publishable plugin package that passes conformance
on day one. Replace the stub body with the real integration and keep the test green."""

from __future__ import annotations

import re
from pathlib import Path

from .plugins import KINDS

STUBS: dict[str, str] = {
    "driver": '''
from typing import Any

import nenyax


class {cls}Driver(nenyax.Driver):
    """Routes `{name}:<target>` URIs to environments in your format."""

    name = "{name}"
    format = "{cls} environments"

    def load(self, target: str, **options: Any) -> nenyax.Environment:
        # TODO: translate your native environment into the Nenyax contract. Subclass
        # nenyax.StepEnvironment (you drive steps) or nenyax.ModelDrivenEnvironment (it calls a
        # model). This stub serves a two-task environment so conformance passes immediately.
        return nenyax.define(
            f"{name}-{{target}}",
            tasks=[{{"prompt": "Say hi in answer tags.", "answer": "hi"}},
                   {{"prompt": "Say yes in answer tags.", "answer": "yes"}}],
            score=lambda t, r: float(r.strip() == f"<answer>{{t['answer']}}</answer>"),
            reference=lambda t: f"<answer>{{t['answer']}}</answer>",
        )
''',
    "sandbox": '''
from nenyax.sandbox import BackendInfo, Isolation, SandboxBackend, SandboxSpec
from nenyax.sandbox.process import ProcessBackend


class {cls}Backend(SandboxBackend):
    """Sandboxes on {cls}."""

    name = "{name}"
    credentials = ()  # e.g. ("{upper}_API_KEY",): bring-your-own-key, never stored
    info = BackendInfo(  # declare only what you really enforce; conformance checks every claim
        isolation=Isolation.PROCESS,
        enforces=frozenset({{"cpu", "timeout"}}),
        fork="disk",
    )

    def create(self, spec: SandboxSpec | None = None):
        # TODO: call your provider's SDK (import it lazily). This stub delegates to the local
        # process backend so conformance passes immediately.
        return ProcessBackend().create(spec)
''',
    "learner": '''
from nenyax import FunctionModel, Trajectory


class {cls}Learner:
    """Turns judged experience into a better policy."""

    def __init__(self, model=None, **options):
        # TODO: connect to your trainer or platform. `model` arrives as a config spec or object.
        self.options = options

    def policy(self):
        # TODO: return a ChatModel (or Endpoint) that acts with your current weights.
        return FunctionModel(lambda messages: "")

    def update(self, groups: list[list[Trajectory]]) -> dict[str, float]:
        # groups = rollouts of the same task; nenyax.learners.group_advantages(groups) helps.
        return {{"groups": float(len(groups))}}

    # Optional, enables train(gate=True) to revert harmful updates:
    # def snapshot(self): ...
    # def restore(self, state): ...
''',
    "judge": '''
from nenyax import Judgment, Trajectory
from nenyax.judges import final_text


class {cls}Judge:
    """Scores episodes your way, on any environment."""

    name = "{name}"

    def __init__(self, **options):
        self.options = options

    def judge(self, traj: Trajectory) -> Judgment:
        # TODO: your scoring. This stub keeps the environment's own verdict.
        base = traj.judgment.score if traj.judgment else 0.0
        return Judgment(score=base, source="verifier", judge=self.name,
                        details={{"final_text": final_text(traj)[:200]}})
''',
    "hub": '''
from nenyax.integrations import Hub, Listing


class {cls}Hub(Hub):
    """Search {cls} for environments; return URIs Nenyax can load."""

    name = "{name}"
    summary = "{cls} environments"
    credentials = ()  # e.g. ("{upper}_TOKEN",)

    def search(self, query: str = "", limit: int = 20) -> list[Listing]:
        # TODO: call your catalog API and map each result to a Nenyax URI
        # (e.g. "openenv:https://...", "verifiers:<module>", "harbor:<path>").
        return []

    def fetch(self, listing: Listing) -> str:
        # TODO: download or install if needed; return the loadable URI.
        return listing.uri
''',
    "telemetry": '''
from nenyax import Trajectory
from nenyax.integrations.telemetry import episode_metrics


class {cls}Telemetry:
    """Sends episode metrics to {cls}."""

    def __init__(self, **options):
        self.options = options
        self.sent = 0

    def on_trajectory(self, traj: Trajectory) -> None:
        metrics = episode_metrics(traj)  # score, ok, steps, model_calls, duration_s, reward/*
        # TODO: send `metrics` (and traj.model_calls, if useful) to your service.
        self.sent += 1
''',
}

TESTS: dict[str, str] = {
    "driver": "assert_conforms(nenyax.load('{name}:demo'), 'audited')",
    "sandbox": "assert_sandbox_conforms({cls}Backend())",
    "learner": "assert_learner_conforms({cls}Learner())",
    "judge": "assert_judge_conforms({cls}Judge())",
    "hub": "assert_hub_conforms({cls}Hub())",
    "telemetry": "assert_callback_conforms({cls}Telemetry())",
}

SUFFIX = {
    "driver": "Driver",
    "sandbox": "Backend",
    "learner": "Learner",
    "judge": "Judge",
    "hub": "Hub",
    "telemetry": "Telemetry",
}


def scaffold(kind: str, name: str, dest: str | Path = ".") -> Path:
    if kind not in STUBS:
        raise ValueError(f"kind must be one of {sorted(STUBS)}")
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    cls = "".join(part.capitalize() for part in slug.split("_"))
    fmt = {"name": slug, "cls": cls, "upper": slug.upper()}
    root = Path(dest) / f"nenyax-{slug.replace('_', '-')}"
    pkg = root / "src" / f"nenyax_{slug}"
    (root / "tests").mkdir(parents=True, exist_ok=False)
    pkg.mkdir(parents=True)
    obj = f"{cls}{SUFFIX[kind]}"
    (pkg / "__init__.py").write_text(
        f'"""{cls} {kind} plugin for Nenyax."""\n' + STUBS[kind].format(**fmt)
    )
    group = KINDS[kind].group
    (root / "pyproject.toml").write_text(f'''[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "nenyax-{slug.replace("_", "-")}"
version = "0.1.0"
description = "{cls} {kind} for Nenyax"
requires-python = ">=3.11"
dependencies = ["nenyax"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.entry-points."{group}"]
{slug} = "nenyax_{slug}:{obj}"

[tool.hatch.build.targets.wheel]
packages = ["src/nenyax_{slug}"]
''')
    test = TESTS[kind].format(**fmt)
    helper = test.split("(", 1)[0]
    (root / "tests" / "test_conformance.py").write_text(f"""import nenyax  # noqa: F401
from nenyax.testing import {helper}

from nenyax_{slug} import {obj}  # noqa: F401


def test_{slug}_conforms():
    {test}
""")
    (root / "README.md").write_text(f"""# nenyax-{slug.replace("_", "-")}

A Nenyax **{kind}** plugin. Installing this package registers it under the
`{group}` entry point as `{slug}`, with no change to Nenyax.

```bash
pip install -e '.[dev]'
pytest                      # nenyax.testing.{helper} must pass
nenyax plugins              # shows `{kind}  {slug}  nenyax-{slug.replace("_", "-")}`
```

To list it in the public registry, open a PR adding it to `registry/plugins.toml` in the Nenyax
repository.
""")
    return root
