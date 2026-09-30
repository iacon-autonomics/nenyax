"""Sandbox conformance: verify a backend does what its ``BackendInfo`` claims.

Every backend, built-in or bring-your-own-key, runs the same suite. A claimed guarantee that
fails (e.g. ``enforces`` includes ``network`` but a connection succeeds) is a failure, not a
warning.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .base import Sandbox, SandboxBackend, SandboxSpec


@dataclass
class SandboxCheck:
    name: str
    status: Literal["pass", "fail", "skip"]
    detail: str = ""
    duration_s: float = 0.0


@dataclass
class SandboxReport:
    backend: str
    checks: list[SandboxCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    def render(self) -> str:
        icon = {"pass": "✔", "fail": "✘", "skip": "–"}
        lines = [f"sandbox backend {self.backend}  →  {'OK' if self.ok else 'FAILED'}"]
        lines += [
            f"  {icon[c.status]} {c.name:<22} {c.duration_s:6.2f}s  {c.detail}" for c in self.checks
        ]
        return "\n".join(lines)


def check_backend(backend: SandboxBackend, spec: SandboxSpec | None = None) -> SandboxReport:
    spec = spec or SandboxSpec()
    info = backend.info
    report = SandboxReport(backend=backend.name)

    def record(name: str, fn: Callable[[], str | None], skip: str | None = None) -> None:
        if skip:
            report.checks.append(SandboxCheck(name, "skip", skip))
            return
        t0 = time.perf_counter()
        try:
            err = fn()
            status = "fail" if err else "pass"
        except Exception as e:
            err, status = f"{type(e).__name__}: {e}", "fail"
        report.checks.append(SandboxCheck(name, status, err or "", time.perf_counter() - t0))

    box: list[Sandbox] = []
    record("create", lambda: box.append(backend.create(spec)))
    if not box:
        return report
    sb = box[0]
    try:
        record("exec", lambda: None if sb.exec("echo hi").stdout.strip() == "hi" else "bad stdout")
        record("exit_code", lambda: None if sb.exec("exit 3").exit_code == 3 else "wrong code")

        def files() -> str | None:
            sb.write_file(f"{spec.workdir}/a/b.txt", "payload")
            if sb.read_file(f"{spec.workdir}/a/b.txt") != b"payload":
                return "read back differs"
            seen = sb.exec("cat a/b.txt").stdout
            if seen != "payload":
                return f"exec cannot see written file: {seen!r}"
            sb.write_file("rel.txt", "relative")  # relative paths resolve against the workdir
            if sb.read_file(f"{spec.workdir}/rel.txt") != b"relative":
                return "relative path not resolved against the workdir"
            big = bytes(range(256)) * 4096  # 1 MiB of every byte value
            sb.write_file("bin.dat", big)
            return None if sb.read_file("bin.dat") == big else "binary round trip corrupted"

        record("files", files)

        def timeout() -> str | None:
            r = sb.exec("sleep 30", timeout_s=2)
            return None if r.timed_out and r.duration_s < 20 else f"not enforced: {r}"

        record("timeout", timeout, skip=None if "timeout" in info.enforces else "not claimed")

        def network() -> str | None:
            has_probe = sb.exec("command -v python3 >/dev/null || command -v wget >/dev/null")
            if not has_probe.ok:
                return "no python3 or wget in the image to probe with; use an image that has one"
            probe = (
                "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\""
                " 2>/dev/null || wget -q -T 3 -O- http://1.1.1.1 >/dev/null 2>&1"
            )
            r = sb.exec(probe, timeout_s=15)
            return "outbound connection succeeded" if r.exit_code == 0 else None

        record(
            "network_blocked",
            network,
            skip=None if "network" in info.enforces and not spec.network else "not claimed",
        )

        def filesystem() -> str | None:
            with tempfile.NamedTemporaryFile("w", suffix=".nenyax", delete=False) as f:
                f.write(marker := uuid.uuid4().hex)
            try:
                r = sb.exec(f"cat {f.name}")
                return "host file readable from sandbox" if marker in r.stdout else None
            finally:
                Path(f.name).unlink(missing_ok=True)

        record(
            "filesystem_contained",
            filesystem,
            skip=None if "filesystem" in info.enforces else "not claimed",
        )

        def fork() -> str | None:
            sb.write_file(f"{spec.workdir}/state.txt", "parent")
            with sb.fork() as child:
                if child.read_file(f"{spec.workdir}/state.txt") != b"parent":
                    return "fork lost files"
                child.write_file(f"{spec.workdir}/state.txt", "child")
            if sb.read_file(f"{spec.workdir}/state.txt") != b"parent":
                return "child write leaked into parent"
            return None

        record("fork", fork, skip=None if info.fork != "none" else "not claimed")
    finally:
        record("destroy", lambda: sb.destroy())
    return report
