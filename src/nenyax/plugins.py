"""Every extension point in one place.

Nenyax is extended by *plugins*: ordinary Python packages that declare entry points. Installing
the package is enough; no change to Nenyax, no registration call::

    # pyproject.toml of acme-nenyax
    [project.entry-points."nenyax.sandboxes"]
    acme = "acme_nenyax:AcmeBackend"

========== ====================== ====================================
kind       entry-point group      conformance test (nenyax.testing)
========== ====================== ====================================
driver     nenyax.drivers         assert_conforms
sandbox    nenyax.sandboxes       assert_sandbox_conforms
provider   nenyax.providers       assert_provider_conforms
hub        nenyax.hubs            assert_hub_conforms
learner    nenyax.learners        assert_learner_conforms
judge      nenyax.judges          assert_judge_conforms
telemetry  nenyax.telemetry       assert_callback_conforms
========== ====================== ====================================

Run ``nenyax plugins --kinds`` for the interface each kind implements, and
``nenyax new-plugin <kind> <name>`` for a package that passes its conformance test on day one.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points


@dataclass(frozen=True)
class Kind:
    name: str
    group: str
    interface: str
    conformance: str


KINDS: dict[str, Kind] = {
    k.name: k
    for k in [
        Kind("driver", "nenyax.drivers", "nenyax.Driver", "assert_conforms"),
        Kind(
            "sandbox",
            "nenyax.sandboxes",
            "nenyax.sandbox.SandboxBackend",
            "assert_sandbox_conforms",
        ),
        Kind(
            "provider",
            "nenyax.providers",
            "nenyax.integrations.Provider",
            "assert_provider_conforms",
        ),
        Kind("hub", "nenyax.hubs", "nenyax.integrations.Hub", "assert_hub_conforms"),
        Kind("learner", "nenyax.learners", "policy() + update(groups)", "assert_learner_conforms"),
        Kind("judge", "nenyax.judges", "judge(traj) -> Judgment", "assert_judge_conforms"),
        Kind("telemetry", "nenyax.telemetry", "on_trajectory(traj)", "assert_callback_conforms"),
    ]
}


@dataclass(frozen=True)
class Plugin:
    kind: str
    name: str
    source: str
    """``builtin`` or the distribution (package) that provides it."""
    target: str


def _builtins() -> list[Plugin]:
    import importlib

    from . import config, integrations, registry, sandbox

    hub_module = importlib.import_module("nenyax.integrations.hubs")  # `hubs` is also a function

    out = [Plugin("driver", n, "builtin", ref) for n, ref in registry._BUILTIN.items()]
    out += [Plugin("sandbox", n, "builtin", ref) for n, ref in sandbox.registry._BUILTIN.items()]
    out += [
        Plugin("provider", n, "builtin", spec.base_url)
        for n, spec in integrations.PROVIDERS.items()
    ]
    out += [
        Plugin("hub", cls.name, "builtin", f"{cls.__module__}:{cls.__name__}")
        for cls in hub_module._BUILTIN_HUBS
    ]
    for kind in ("learner", "judge", "telemetry"):
        out += [Plugin(kind, n, "builtin", "nenyax") for n in config.BUILTIN[kind]]
    return out


def installed() -> list[Plugin]:
    """Built-in and third-party plugins, third-party last (they override built-ins by name)."""
    plugins = _builtins()
    for kind in KINDS.values():
        for ep in entry_points(group=kind.group):
            dist = ep.dist.name if getattr(ep, "dist", None) else "?"
            if dist == "nenyax":
                continue
            plugins.append(Plugin(kind.name, ep.name, dist, ep.value))
    return plugins
