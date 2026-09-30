"""The sandbox contract: the few operations every isolation backend provides.

Every provider we surveyed (Docker, microVMs, and hosted services such as E2B, Daytona, Modal or
Prime) converges on the same core: create, exec, files, destroy. Snapshot, fork and port exposure
are optional and declared, never assumed.

A backend is a class. Built-in backends ship with Nenyax. Bring-your-own-key hosted providers
live in their own modules or packages and declare the credentials they need.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from ..environment import NenyaxError, UnsupportedCapability
from ..types import Isolation


class SandboxError(NenyaxError):
    pass


class BackendNotConfigured(SandboxError):
    """A backend needs credentials or software that are not present."""


@dataclass(frozen=True)
class SandboxSpec:
    """What an environment asks for. Backends honour what they can and declare the rest."""

    image: str = "python:3.12-slim"
    cpus: float = 1.0
    memory_mb: int = 1024
    timeout_s: float = 3600.0
    """Lifetime of the sandbox."""
    network: bool = False
    """Outbound network access. Off by default: most RL actions should not reach the internet."""
    env: dict[str, str] = field(default_factory=dict)
    ports: tuple[int, ...] = ()
    """Ports inside the sandbox to make reachable from the host."""
    workdir: str = "/workspace"
    output_cap: int = 1 << 20
    """Bytes of stdout and of stderr kept per ``exec``; the rest is drained and dropped."""


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    """Killed because it exceeded its time limit."""
    duration_s: float = 0.0
    truncated: bool = False
    """Output exceeded the capture cap and was cut (the command itself was not affected)."""
    killed: bool = False
    """Killed by a signal for another reason, typically the memory limit (OOM)."""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.killed


@dataclass(frozen=True)
class BackendInfo:
    """What a backend can honestly do."""

    isolation: Isolation
    enforces: frozenset[Literal["cpu", "memory", "network", "filesystem", "timeout"]]
    """Limits the backend actually enforces (vs. merely requests)."""
    fork: Literal["none", "disk", "memory"] = "none"
    """``disk``: files survive a fork, running processes don't. ``memory``: full live state."""
    ports: bool = False
    gpu: bool = False


class Sandbox(ABC):
    """One live, isolated workspace. Use as a context manager."""

    id: str
    spec: SandboxSpec

    @abstractmethod
    def exec(
        self,
        command: str,
        *,
        timeout_s: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a shell command inside the sandbox."""

    @abstractmethod
    def write_file(self, path: str, data: bytes | str) -> None: ...

    @abstractmethod
    def read_file(self, path: str) -> bytes: ...

    @abstractmethod
    def destroy(self) -> None: ...

    def fork(self) -> Sandbox:
        """A new sandbox starting from this one's current state (see ``BackendInfo.fork``)."""
        raise UnsupportedCapability(f"{type(self).__name__} cannot fork")

    def url(self, port: int) -> str:
        """Host-reachable URL for a port listed in ``spec.ports``."""
        raise UnsupportedCapability(f"{type(self).__name__} cannot expose ports")

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.destroy()


class SandboxBackend(ABC):
    """Creates sandboxes. Subclass this to add a provider."""

    #: Registry name, e.g. ``"docker"``.
    name: str
    info: BackendInfo
    #: Environment variables that must be set (bring-your-own-key hosted providers).
    credentials: tuple[str, ...] = ()

    def __init__(self, **options: str) -> None:
        """Options override credential environment variables, e.g. ``api_key=...``."""
        self.options = options

    def credential(self, var: str) -> str | None:
        return self.options.get(var.lower()) or os.environ.get(var)

    def missing(self) -> list[str]:
        """What prevents this backend from running here (software or credentials)."""
        return [f"set {v}" for v in self.credentials if not self.credential(v)]

    def available(self) -> bool:
        return not self.missing()

    @abstractmethod
    def create(self, spec: SandboxSpec | None = None) -> Sandbox: ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} isolation={self.info.isolation}>"
