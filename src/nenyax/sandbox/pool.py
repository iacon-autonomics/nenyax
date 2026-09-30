"""Warm pools: pay setup once, hand out fresh sandboxes per episode.

RL runs thousands of short episodes on the same starting state. A pool prepares one *template*
sandbox (image + setup commands), then serves each episode a fork of it when the backend can
fork, or a newly created sandbox otherwise. Every episode starts from identical state.
"""

from __future__ import annotations

import threading
from collections import deque

from .base import Sandbox, SandboxBackend, SandboxError, SandboxSpec


class WarmPool:
    def __init__(
        self,
        backend: SandboxBackend,
        spec: SandboxSpec | None = None,
        *,
        setup: list[str] | None = None,
        size: int = 4,
    ) -> None:
        """
        Args:
            setup: shell commands run once in the template (install deps, seed files).
            size: sandboxes kept ready.
        """
        self.backend, self.spec, self.size = backend, spec or SandboxSpec(), size
        self.setup = list(setup or [])
        self._ready: deque[Sandbox] = deque()
        self._lock = threading.Lock()
        self.template = self._prepare(backend.create(self.spec))
        self.fill()

    def _prepare(self, sandbox: Sandbox) -> Sandbox:
        for command in self.setup:
            result = sandbox.exec(command)
            if not result.ok:
                sandbox.destroy()
                raise SandboxError(f"pool setup failed on {command!r}: {result.stderr[:300]}")
        return sandbox

    def _fresh(self) -> Sandbox:
        if self.backend.info.fork != "none":
            return self.template.fork()
        return self._prepare(self.backend.create(self.spec))

    def fill(self) -> None:
        while len(self._ready) < self.size:
            sandbox = self._fresh()
            with self._lock:
                self._ready.append(sandbox)

    def acquire(self) -> Sandbox:
        """A sandbox in the template's state. Destroy it (or use ``with``) when done."""
        with self._lock:
            sandbox = self._ready.popleft() if self._ready else None
        return sandbox or self._fresh()

    def close(self) -> None:
        with self._lock:
            ready, self._ready = list(self._ready), deque()
        for sandbox in [*ready, self.template]:
            sandbox.destroy()

    def __enter__(self) -> WarmPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
