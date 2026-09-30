"""Isolation for untrusted actions: one small contract, pluggable backends::

    from nenyax import sandbox

    backend = sandbox.select("container")  # lightest ready backend that is safe enough
    with backend.create(sandbox.SandboxSpec(image="python:3.12-slim")) as sb:
        print(sb.exec("python -c 'print(6*7)'").stdout)

    with sandbox.WarmPool(backend, setup=["pip install numpy"], size=8) as pool:
        with pool.acquire() as sb:  # identical fresh state every episode
            ...

Providers plug in through the ``nenyax.sandboxes`` entry-point group; see
docs/writing-a-sandbox-backend.md.
"""

from .base import (
    BackendInfo,
    BackendNotConfigured,
    ExecResult,
    Isolation,
    Sandbox,
    SandboxBackend,
    SandboxError,
    SandboxSpec,
)
from .conformance import SandboxReport, check_backend
from .pool import WarmPool
from .registry import backends, get, register, select

__all__ = [
    "BackendInfo",
    "BackendNotConfigured",
    "ExecResult",
    "Isolation",
    "Sandbox",
    "SandboxBackend",
    "SandboxError",
    "SandboxReport",
    "SandboxSpec",
    "WarmPool",
    "backends",
    "check_backend",
    "get",
    "register",
    "select",
]
