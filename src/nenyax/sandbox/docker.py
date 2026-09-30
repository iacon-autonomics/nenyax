"""``docker``: one container per sandbox, driven through the Docker (or Podman) CLI.

No Python SDK dependency: the ``docker`` binary is the contract, and ``podman`` is a drop-in
(``DockerBackend(binary="podman")``). Networking is off unless the spec asks for it. Forks are
disk-level: the container is committed to an image and a new container starts from it.
"""

from __future__ import annotations

import posixpath
import shlex
import shutil
import subprocess
import time
import uuid

from ..environment import UnsupportedCapability
from ._io import run_bounded
from .base import (
    BackendInfo,
    ExecResult,
    Isolation,
    Sandbox,
    SandboxBackend,
    SandboxError,
    SandboxSpec,
)

LABEL = "nenyax.sandbox=1"


class DockerSandbox(Sandbox):
    def __init__(self, backend: DockerBackend, spec: SandboxSpec, image: str | None = None) -> None:
        if spec.ports and not spec.network:
            # Docker cannot publish ports from a container with networking disabled; refuse
            # rather than silently opening the network.
            raise UnsupportedCapability(
                "docker: SandboxSpec.ports requires network=True (no-network containers "
                "cannot publish ports)"
            )
        self.backend, self.spec = backend, spec
        self._images: list[str] = []
        args = [
            "run",
            "-d",
            "--label",
            LABEL,
            "--cpus",
            str(spec.cpus),
            "--memory",
            f"{spec.memory_mb}m",
            "--pids-limit",
            "512",
            "-w",
            spec.workdir,
        ]
        if not spec.network:
            args += ["--network", "none"]
        for port in spec.ports:
            args += ["-p", f"127.0.0.1::{port}"]
        for key, value in spec.env.items():
            args += ["-e", f"{key}={value}"]
        boot = f"mkdir -p {shlex.quote(spec.workdir)}; exec sleep infinity"
        args += [image or spec.image, "sh", "-c", boot]
        self.id = backend.run(args).stdout.strip()

    def exec(self, command, *, timeout_s=None, cwd=None, env=None) -> ExecResult:
        limit = timeout_s or self.spec.timeout_s
        args = [self.backend.binary, "exec", "-w", cwd or self.spec.workdir]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        # `timeout` runs inside the container so the command dies there, not just our client.
        # It exits 124 on timeout; a bare 137 therefore means another SIGKILL, usually OOM.
        seconds = str(max(1, int(limit)))
        args += [self.id, "timeout", "-k", "2", seconds, "sh", "-c", command]
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        cap = run_bounded(proc, timeout=limit + 30, kill=proc.kill, cap=self.spec.output_cap)
        timed_out = cap.timed_out or cap.returncode == 124
        return ExecResult(
            exit_code=cap.returncode,
            stdout=cap.stdout.decode(errors="replace"),
            stderr=cap.stderr.decode(errors="replace"),
            timed_out=timed_out,
            duration_s=time.perf_counter() - t0,
            truncated=cap.truncated,
            killed=cap.returncode == 137 and not timed_out,
        )

    def _abs(self, path: str) -> str:
        """Relative paths are relative to the sandbox workdir, as for ``exec``."""
        return posixpath.normpath(posixpath.join(self.spec.workdir, path))

    def write_file(self, path: str, data: bytes | str) -> None:
        payload = data.encode() if isinstance(data, str) else data
        target = self._abs(path)
        script = f"mkdir -p {shlex.quote(posixpath.dirname(target))} && cat > {shlex.quote(target)}"
        self.backend.run(["exec", "-i", self.id, "sh", "-c", script], input=payload)

    def read_file(self, path: str) -> bytes:
        proc = self.backend.run(["exec", self.id, "cat", self._abs(path)], text=False, check=False)
        if proc.returncode != 0:
            raise FileNotFoundError(f"{path} in {self.id}: {proc.stderr.strip()}")
        return proc.stdout

    def fork(self) -> DockerSandbox:
        image = f"nenyax-fork:{uuid.uuid4().hex[:12]}"
        self.backend.run(["commit", self.id, image])
        child = DockerSandbox(self.backend, self.spec, image=image)
        child._images.append(image)
        return child

    def url(self, port: int) -> str:
        if port not in self.spec.ports:
            raise ValueError(f"port {port} was not requested in SandboxSpec.ports")
        mapping = self.backend.run(["port", self.id, str(port)]).stdout.split()[0]
        host, _, published = mapping.rpartition(":")
        return f"http://{self.backend.advertise_host or host}:{published}"

    def destroy(self) -> None:
        self.backend.run(["rm", "-f", self.id], check=False)
        for image in self._images:
            self.backend.run(["rmi", "-f", image], check=False)


class DockerBackend(SandboxBackend):
    name = "docker"
    info = BackendInfo(
        isolation=Isolation.CONTAINER,
        enforces=frozenset({"cpu", "memory", "network", "filesystem", "timeout"}),
        fork="disk",
        ports=True,
    )

    def __init__(
        self, binary: str = "docker", advertise_host: str | None = None, **options: str
    ) -> None:
        """
        Args:
            binary: ``docker`` or ``podman``.
            advertise_host: host name clients should use for exposed ports. Set it to
                ``host.docker.internal`` when the controller itself runs in a container.
        """
        super().__init__(**options)
        self.binary, self.advertise_host = binary, advertise_host

    def missing(self) -> list[str]:
        if not shutil.which(self.binary):
            return [f"install {self.binary}"]
        probe = subprocess.run([self.binary, "info"], capture_output=True)
        return [] if probe.returncode == 0 else [f"start the {self.binary} daemon"]

    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        timeout: float | None = None,
        input: bytes | None = None,
        text: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run ``<binary> *args``; stdout is str (or bytes with ``text=False``), stderr is str."""
        proc = subprocess.run(
            [self.binary, *args], capture_output=True, timeout=timeout, input=input
        )
        stdout = proc.stdout.decode(errors="replace") if text else proc.stdout
        result = subprocess.CompletedProcess(
            proc.args, proc.returncode, stdout, proc.stderr.decode(errors="replace")
        )
        if check and result.returncode != 0:
            raise SandboxError(f"{self.binary} {args[0]} failed: {result.stderr[:300]}")
        return result

    def create(self, spec: SandboxSpec | None = None) -> DockerSandbox:
        return DockerSandbox(self, spec or SandboxSpec())

    def cleanup(self) -> int:
        """Remove every container Nenyax created (e.g. after a crash). Returns the count."""
        ids = self.run(["ps", "-aq", "--filter", f"label={LABEL}"]).stdout.split()
        if ids:
            self.run(["rm", "-f", *ids], check=False)
        return len(ids)
