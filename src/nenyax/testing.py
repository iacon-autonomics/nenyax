"""Helpers for driver and sandbox-backend authors::

    from nenyax.testing import assert_conforms, assert_sandbox_conforms

    def test_my_driver():
        assert_conforms(nenyax.load("mydriver:some-env"), "trainable")

    def test_my_backend():
        assert_sandbox_conforms(MyBackend(api_key="..."))

Both raise ``AssertionError`` with the full report when a claim does not hold.
"""

from __future__ import annotations

from typing import Any

from .conformance import Report, Tier, check
from .environment import Environment
from .sandbox import SandboxBackend, SandboxReport, check_backend


def assert_conforms(env: Environment, tier: str | Tier, **check_kwargs: Any) -> Report:
    want = tier if isinstance(tier, Tier) else Tier[tier.upper()]
    report = check(env, **check_kwargs)
    if report.tier < want:
        raise AssertionError(f"expected at least {want.label}, got:\n{report.render()}")
    return report


def assert_sandbox_conforms(backend: SandboxBackend, **check_kwargs: Any) -> SandboxReport:
    """Fail unless ``backend`` delivers everything its ``BackendInfo`` claims."""
    report = check_backend(backend, **check_kwargs)
    if not report.ok:
        raise AssertionError(report.render())
    return report


# -- conformance for every plugin kind -----------------------------------------------------------


def _toy_trajectories():
    from .define import define
    from .policy import FunctionModel

    env = define(
        "conformance-toy",
        tasks=[{"prompt": f"Say {i} in answer tags.", "answer": str(i)} for i in range(4)],
        score=lambda t, r: float(r.strip() == f"<answer>{t['answer']}</answer>"),
    )
    right = FunctionModel(lambda m: f"<answer>{m[-1].content.split()[1]}</answer>")
    wrong = FunctionModel(lambda m: "no")
    return env, [[env.rollout(right, task=t), env.rollout(wrong, task=t)] for t in env.tasks()]


def assert_judge_conforms(judge: Any) -> None:
    """``judge(traj)`` returns a finite Judgment for successful and failed episodes alike."""
    import math

    from .types import Judgment

    _, groups = _toy_trajectories()
    for traj in (t for g in groups for t in g):
        j = judge.judge(traj)
        assert isinstance(j, Judgment) and math.isfinite(j.score), j


def assert_learner_conforms(learner: Any) -> None:
    """``policy()`` can act, ``update(groups)`` accepts judged groups and returns float metrics."""
    env, groups = _toy_trajectories()
    traj = env.rollout(learner.policy(), task=env.tasks()[0])
    assert traj.error is None, traj.error
    metrics = learner.update(groups)
    assert isinstance(metrics, dict) and all(isinstance(v, (int, float)) for v in metrics.values())
    snap = getattr(learner, "snapshot", None)
    if callable(snap):
        learner.restore(snap())


def assert_callback_conforms(callback: Any) -> None:
    _, groups = _toy_trajectories()
    for traj in (t for g in groups for t in g):
        callback.on_trajectory(traj)


def assert_hub_conforms(hub: Any, query: str = "") -> None:
    """``search`` returns Listings whose URIs route to a known driver."""
    from .registry import drivers

    hits = hub.search(query, limit=3)
    assert isinstance(hits, list)
    for hit in hits:
        assert hit.uri.split(":", 1)[0] in drivers(), f"{hit.uri} routes to no driver"


def assert_provider_conforms(provider: Any, model: str, *, live: bool = False) -> None:
    """The provider yields an Endpoint; with ``live=True`` it must also answer one request."""
    from .types import Message

    endpoint = provider.endpoint(model)
    assert endpoint.base_url.startswith(("http://", "https://")), endpoint.base_url
    if live:
        reply = endpoint.complete([Message(role="user", content="Reply with the word ok.")])
        assert reply.role == "assistant" and reply.content
