"""``process``: a private working directory and an OS process with resource limits.

The lightest real sandbox. It stops runaway loops and fork bombs and keeps files in a scratch
directory, but it shares the host's filesystem view and network, so use it only for actions you
would run on your own machine (and declare ``Isolation.PROCESS``, not more).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from ._io import run_bounded
from .base import BackendInfo, ExecResult, Isolation, Sandbox, SandboxBackend, SandboxSpec


def _limits(spec: SandboxSpec):
    """Build a preexec hook that applies POSIX rlimits in the child."""

    def apply() -> None:
        import resource

        cpu = max(1, int(spec.timeout_s))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        if sys.platform.startswith("linux"):  # macOS ignores address-space limits
            mem = spec.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    return apply


class ProcessSandbox(Sandbox):
    def __init__(self, spec: SandboxSpec) -> None:
        self.id = f"proc-{uuid.uuid4().hex[:10]}"
        self.spec = spec
        self.root = Path(tempfile.mkdtemp(prefix="nenyax-"))

    def _path(self, path: str) -> Path:
        """Map sandbox paths (absolute under ``spec.workdir`` or relative) into the root."""
        rel = path.removeprefix(self.spec.workdir).lstrip("/")
        target = (self.root / rel).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise PermissionError(f"{path!r} escapes the sandbox")
        return target

    def exec(self, command, *, timeout_s=None, cwd=None, env=None) -> ExecResult:
        t0 = time.perf_counter()
        full_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            **self.spec.env,
            **(env or {}),
        }
        proc = subprocess.Popen(
            ["/bin/sh", "-c", command],
            cwd=self._path(cwd) if cwd else self.root,
            env=full_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # own process group, so a timeout kills every child
            preexec_fn=_limits(self.spec),
        )

        def kill() -> None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)

        cap = run_bounded(
            proc, timeout=timeout_s or self.spec.timeout_s, kill=kill, cap=self.spec.output_cap
        )
        return ExecResult(
            exit_code=cap.returncode,
            stdout=cap.stdout.decode(errors="replace"),
            stderr=cap.stderr.decode(errors="replace"),
            timed_out=cap.timed_out,
            duration_s=time.perf_counter() - t0,
            truncated=cap.truncated,
            killed=cap.returncode < 0 and not cap.timed_out,
        )

    def write_file(self, path: str, data: bytes | str) -> None:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.encode() if isinstance(data, str) else data)

    def read_file(self, path: str) -> bytes:
        return self._path(path).read_bytes()

    def fork(self) -> ProcessSandbox:
        child = ProcessSandbox(self.spec)
        shutil.copytree(self.root, child.root, dirs_exist_ok=True, symlinks=True)
        return child

    def destroy(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class ProcessBackend(SandboxBackend):
    name = "process"
    info = BackendInfo(
        isolation=Isolation.PROCESS,
        enforces=frozenset(
            {"cpu", "timeout"} | ({"memory"} if sys.platform.startswith("linux") else set())
        ),
        fork="disk",
    )

    def missing(self) -> list[str]:
        return ["POSIX only"] if os.name != "posix" else super().missing()

    def create(self, spec: SandboxSpec | None = None) -> ProcessSandbox:
        return ProcessSandbox(spec or SandboxSpec())
