"""Find sandbox backends and pick the lightest one that is safe enough.

Backends come from three places, in order: built-ins, the ``nenyax.sandboxes`` entry-point group
(how packaged and proprietary providers plug in), and :func:`register` at runtime::

    [project.entry-points."nenyax.sandboxes"]
    acme = "acme_nenyax:AcmeBackend"
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Literal

from .base import BackendNotConfigured, Isolation, SandboxBackend

_BUILTIN = {
    "process": "nenyax.sandbox.process:ProcessBackend",
    "docker": "nenyax.sandbox.docker:DockerBackend",
    # Hosted, bring your own key (nenyax.sandbox.hosted): ready once the package and key exist.
    "e2b": "nenyax.sandbox.hosted:E2BBackend",
    "daytona": "nenyax.sandbox.hosted:DaytonaBackend",
    "modal": "nenyax.sandbox.hosted:ModalBackend",
    "runloop": "nenyax.sandbox.hosted:RunloopBackend",
}

_FORK_RANK = {"none": 0, "disk": 1, "memory": 2}


def _load(ref: str) -> type[SandboxBackend]:
    module, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module), attr)


@lru_cache(maxsize=1)
def _classes() -> dict[str, type[SandboxBackend]]:
    found = {name: _load(ref) for name, ref in _BUILTIN.items()}
    for ep in entry_points(group="nenyax.sandboxes"):
        found[ep.name] = ep.load()
    return found


def register(backend_cls: type[SandboxBackend]) -> None:
    _classes()[backend_cls.name] = backend_cls


def backends() -> dict[str, type[SandboxBackend]]:
    return dict(_classes())


def get(name: str, **options: str) -> SandboxBackend:
    """Instantiate a backend by name; options may carry credentials (``api_key=...``)."""
    try:
        backend = _classes()[name](**options)
    except KeyError:
        known = sorted(_classes())
        raise BackendNotConfigured(f"unknown sandbox backend {name!r}; have {known}") from None
    missing = backend.missing()
    if missing:
        raise BackendNotConfigured(f"sandbox backend {name!r} is not ready: {'; '.join(missing)}")
    return backend


def select(
    isolation: Isolation | str,
    *,
    fork: Literal["none", "disk", "memory"] = "none",
    ports: bool = False,
    prefer: list[str] | None = None,
) -> SandboxBackend | None:
    """The lightest ready backend meeting the requirements, or ``None`` if none is needed.

    ``prefer`` lists backend names to try first (e.g. a paid provider you have a key for).
    """
    required = Isolation(isolation)
    if required is Isolation.NONE and fork == "none" and not ports:
        return None
    candidates = []
    for name, cls in _classes().items():
        info = cls.info
        if not info.isolation.satisfies(required):
            continue
        if _FORK_RANK[info.fork] < _FORK_RANK[fork] or (ports and not info.ports):
            continue
        preferred = (prefer or []).index(name) if name in (prefer or []) else len(prefer or [])
        candidates.append((preferred, info.isolation.rank, name, cls))
    for *_, cls in sorted(candidates):
        backend = cls()
        if backend.available():
            return backend
    raise BackendNotConfigured(
        f"no ready sandbox backend offers isolation>={required}, fork>={fork}, ports={ports}"
    )
