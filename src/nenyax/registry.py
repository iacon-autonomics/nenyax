"""The router: resolve an environment URI to a driver and load it.

URIs look like ``<driver>:<target>[?key=value&...]``::

    gymnasium:CartPole-v1
    reasoning_gym:basic_arithmetic?size=20&seed=1
    verifiers:my_env_module
    harbor:./tasks/hello-world

A bare local path is routed to whichever driver recognizes it (e.g. a directory containing
``task.toml`` goes to Harbor). Third-party packages add drivers through the ``nenyax.drivers``
entry-point group; no change to Nenyax is needed.
"""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from functools import lru_cache
from importlib.metadata import entry_points
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from .environment import Environment, NenyaxError


class DriverNotFound(NenyaxError):
    pass


class DriverUnavailable(NenyaxError):
    """The driver exists but its native package is not installed."""


class Driver(ABC):
    """Translates one native format into the Nenyax contract."""

    #: URI scheme, e.g. ``"gymnasium"``.
    name: str
    #: Human-readable native format name.
    format: str
    #: Importable module of the native package; used to report availability.
    module: str | None = None
    #: ``pip install nenyax[<extra>]`` that provides the native package.
    extra: str | None = None

    @abstractmethod
    def load(self, target: str, **options: Any) -> Environment: ...

    def detect(self, path: Path) -> bool:
        """Return True if ``path`` is in this driver's native format."""
        return False

    def available(self) -> bool:
        return self.module is None or find_spec(self.module) is not None

    def require(self, module: str) -> Any:
        try:
            return importlib.import_module(module)
        except ImportError as e:
            hint = f"pip install 'nenyax[{self.extra}]'" if self.extra else f"install {module}"
            raise DriverUnavailable(f"driver '{self.name}' needs {module!r}: {hint}") from e


_BUILTIN: dict[str, str] = {
    "gymnasium": "nenyax.formats.gymnasium:GymnasiumDriver",
    "reasoning_gym": "nenyax.formats.reasoning_gym:ReasoningGymDriver",
    "openenv": "nenyax.formats.openenv:OpenEnvDriver",
    "harbor": "nenyax.formats.harbor:HarborDriver",
    "verifiers": "nenyax.formats.verifiers:VerifiersDriver",
    "nemo_gym": "nenyax.formats.nemo_gym:NemoGymDriver",
    "nenyax": "nenyax.formats.nenyax_hub:NenyaxHubDriver",
}


def _instantiate(ref: str) -> Driver:
    module, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module), attr)()


@lru_cache(maxsize=1)
def drivers() -> dict[str, Driver]:
    """All known drivers: built-in first, then entry-point plugins (which may override)."""
    found = {name: _instantiate(ref) for name, ref in _BUILTIN.items()}
    for ep in entry_points(group="nenyax.drivers"):
        found[ep.name] = ep.load()()
    return found


def register(driver: Driver) -> None:
    """Register a driver at runtime (useful in notebooks and tests)."""
    drivers()[driver.name] = driver


def _coerce(value: str) -> Any:
    try:
        return json.loads(value)
    except ValueError:
        return value


def parse_uri(uri: str) -> tuple[str | None, str, dict[str, Any]]:
    """Split ``driver:target?k=v`` into ``(driver, target, options)``."""
    base, _, query = uri.partition("?")
    options = {k: _coerce(v) for k, v in parse_qsl(query, keep_blank_values=True)}
    scheme, sep, target = base.partition(":")
    if sep and scheme in drivers():
        return scheme, target, options
    return None, base, options


def load(uri: str, **options: Any) -> Environment:
    """Load any environment by URI. Keyword options override query-string options."""
    name, target, query = parse_uri(uri)
    query.update(options)
    if name is None:
        path = Path(target).expanduser()
        if (path / "nenyax.toml").is_file():  # a folder made with `nenyax new-env`
            from .platform import load_folder

            env = load_folder(path)
            env.manifest.extra.setdefault("uri", str(path.resolve()))
            return env
        if path.exists():
            for driver in drivers().values():
                if driver.detect(path):
                    return driver.load(str(path), **query)
        known = ", ".join(sorted(drivers()))
        raise DriverNotFound(f"cannot route {uri!r}; use '<driver>:<target>' with one of: {known}")
    env = drivers()[name].load(target, **query)
    # Remember how it was loaded, so another process (an external trainer) can rebuild it.
    if not options:
        env.manifest.extra.setdefault("uri", uri)
    return env
