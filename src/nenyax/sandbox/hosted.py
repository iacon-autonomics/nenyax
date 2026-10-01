"""Hosted sandbox providers, bring your own key: E2B, Daytona, Modal and Runloop.

Each is a thin adapter from the provider's own SDK to the four sandbox operations (exec, write,
read, destroy). They are gated twice: the provider's package must be installed and its key set
(``E2B_API_KEY``, ``DAYTONA_API_KEY``, Modal's token, ``RUNLOOP_API_KEY``). The Nenyax platform
delivers those keys from your connections, only to the runner executing your job.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import time
import uuid

from .base import (
    BackendInfo,
    ExecResult,
    Isolation,
    Sandbox,
    SandboxBackend,
    SandboxSpec,
)


def _need(module: str, package: str) -> list[str]:
    return [] if importlib.util.find_spec(module) else [f"pip install {package}"]


def _wrap(command: str, cwd: str | None, env: dict[str, str] | None) -> str:
    """One shell command with its own environment and working directory, for SDKs that take a
    plain string."""
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
    body = f"cd {shlex.quote(cwd)} && {command}" if cwd else command
    return f"{prefix} sh -c {shlex.quote(body)}" if prefix else f"sh -c {shlex.quote(body)}"


def _abs(spec: SandboxSpec, path: str) -> str:
    return path if path.startswith("/") else f"{spec.workdir.rstrip('/')}/{path}"


def _result(code: int, out: str, err: str, t0: float, cap: int, timed_out=False) -> ExecResult:
    truncated = len(out) > cap or len(err) > cap
    return ExecResult(
        exit_code=code,
        stdout=out[:cap],
        stderr=err[:cap],
        timed_out=timed_out,
        duration_s=time.perf_counter() - t0,
        truncated=truncated,
    )


HOSTED = BackendInfo(
    isolation=Isolation.MICROVM,
    enforces=frozenset({"cpu", "memory", "filesystem", "timeout"}),
    fork="none",
    ports=True,
)


# -- E2B -----------------------------------------------------------------------------------------


class E2BSandbox(Sandbox):
    def __init__(self, backend: E2BBackend, spec: SandboxSpec) -> None:
        from e2b import Sandbox as E2B

        self.spec = spec
        template = backend.options.get("template")
        self._sb = E2B.create(
            template=template,
            timeout=int(spec.timeout_s),
            envs=dict(spec.env),
            api_key=backend.credential("E2B_API_KEY"),
            allow_internet_access=spec.network,
        )
        self.id = f"e2b-{self._sb.sandbox_id}"
        self._sb.commands.run(f"mkdir -p {shlex.quote(spec.workdir)}")

    def exec(self, command, *, timeout_s=None, cwd=None, env=None) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        t0 = time.perf_counter()
        try:
            r = self._sb.commands.run(
                command,
                cwd=cwd or self.spec.workdir,
                envs=env or {},
                timeout=timeout_s or self.spec.timeout_s,
            )
            return _result(r.exit_code, r.stdout, r.stderr, t0, self.spec.output_cap)
        except CommandExitException as e:  # E2B raises on a non-zero exit
            return _result(e.exit_code, e.stdout, e.stderr, t0, self.spec.output_cap)
        except TimeoutError:
            return _result(124, "", "timed out", t0, self.spec.output_cap, timed_out=True)

    def write_file(self, path: str, data: bytes | str) -> None:
        self._sb.files.write(_abs(self.spec, path), data)

    def read_file(self, path: str) -> bytes:
        return bytes(self._sb.files.read(_abs(self.spec, path), format="bytes"))

    def url(self, port: int) -> str:
        return f"https://{self._sb.get_host(port)}"

    def destroy(self) -> None:
        self._sb.kill()


class E2BBackend(SandboxBackend):
    name = "e2b"
    info = HOSTED
    credentials = ("E2B_API_KEY",)

    def missing(self) -> list[str]:
        return _need("e2b", "e2b") + super().missing()

    def create(self, spec: SandboxSpec | None = None) -> E2BSandbox:
        return E2BSandbox(self, spec or SandboxSpec())


# -- Daytona -------------------------------------------------------------------------------------


class DaytonaSandbox(Sandbox):
    def __init__(self, backend: DaytonaBackend, spec: SandboxSpec) -> None:
        from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, Resources

        self.spec = spec
        cfg = DaytonaConfig(
            api_key=backend.credential("DAYTONA_API_KEY"),
            api_url=backend.options.get("api_url") or os.environ.get("DAYTONA_API_URL"),
        )
        self._client = Daytona(cfg)
        params = CreateSandboxFromImageParams(
            image=spec.image,
            env_vars=dict(spec.env),
            resources=Resources(cpu=max(1, int(spec.cpus)), memory=max(1, spec.memory_mb // 1024)),
            auto_stop_interval=max(1, int(spec.timeout_s // 60)),
            network_block_all=not spec.network,
        )
        self._sb = self._client.create(params, timeout=300)
        self.id = f"daytona-{self._sb.id}"
        self._sb.process.exec(f"mkdir -p {shlex.quote(spec.workdir)}")

    def exec(self, command, *, timeout_s=None, cwd=None, env=None) -> ExecResult:
        t0 = time.perf_counter()
        r = self._sb.process.exec(
            command,
            cwd=cwd or self.spec.workdir,
            env=env or None,
            timeout=int(timeout_s or self.spec.timeout_s),
        )
        # Daytona returns combined output in ``result``.
        return _result(int(r.exit_code), r.result or "", "", t0, self.spec.output_cap)

    def write_file(self, path: str, data: bytes | str) -> None:
        payload = data.encode() if isinstance(data, str) else data
        self._sb.fs.upload_file(payload, _abs(self.spec, path))

    def read_file(self, path: str) -> bytes:
        return bytes(self._sb.fs.download_file(_abs(self.spec, path)))

    def url(self, port: int) -> str:
        return self._sb.get_preview_link(port).url

    def destroy(self) -> None:
        self._client.delete(self._sb)


class DaytonaBackend(SandboxBackend):
    name = "daytona"
    info = HOSTED
    credentials = ("DAYTONA_API_KEY",)

    def missing(self) -> list[str]:
        return _need("daytona", "daytona") + super().missing()

    def create(self, spec: SandboxSpec | None = None) -> DaytonaSandbox:
        return DaytonaSandbox(self, spec or SandboxSpec())


# -- Modal ---------------------------------------------------------------------------------------


class ModalSandbox(Sandbox):
    def __init__(self, backend: ModalBackend, spec: SandboxSpec) -> None:
        import modal

        self.spec = spec
        app = modal.App.lookup(
            backend.options.get("app", "nenyax-sandboxes"), create_if_missing=True
        )
        self._sb = modal.Sandbox.create(
            app=app,
            image=modal.Image.from_registry(spec.image),
            timeout=int(spec.timeout_s),
            cpu=spec.cpus,
            memory=spec.memory_mb,
            block_network=not spec.network,
            secrets=[modal.Secret.from_dict(dict(spec.env))] if spec.env else [],
            workdir=spec.workdir,
        )
        self.id = f"modal-{self._sb.object_id}"

    def exec(self, command, *, timeout_s=None, cwd=None, env=None) -> ExecResult:
        t0 = time.perf_counter()
        proc = self._sb.exec(
            "sh", "-c", _wrap(command, cwd, env), timeout=int(timeout_s or self.spec.timeout_s)
        )
        proc.wait()
        out, err = proc.stdout.read(), proc.stderr.read()
        return _result(proc.returncode, out, err, t0, self.spec.output_cap)

    def write_file(self, path: str, data: bytes | str) -> None:
        target = _abs(self.spec, path)
        self._sb.exec("mkdir", "-p", target.rsplit("/", 1)[0]).wait()
        with self._sb.open(target, "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)

    def read_file(self, path: str) -> bytes:
        with self._sb.open(_abs(self.spec, path), "rb") as f:
            return f.read()

    def destroy(self) -> None:
        self._sb.terminate()


class ModalBackend(SandboxBackend):
    name = "modal"
    info = HOSTED
    credentials = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")

    def __init__(self, **options: str) -> None:
        super().__init__(**options)
        # The platform stores a Modal token as one secret, "id:secret" (see Connections).
        packed = self.credential("MODAL_TOKEN_SECRET") or ""
        if ":" in packed and not self.credential("MODAL_TOKEN_ID"):
            token_id, secret = packed.split(":", 1)
            os.environ.setdefault("MODAL_TOKEN_ID", token_id)
            os.environ["MODAL_TOKEN_SECRET"] = secret

    def missing(self) -> list[str]:
        return _need("modal", "modal") + super().missing()

    def create(self, spec: SandboxSpec | None = None) -> ModalSandbox:
        return ModalSandbox(self, spec or SandboxSpec())


# -- Runloop -------------------------------------------------------------------------------------


class RunloopSandbox(Sandbox):
    def __init__(self, backend: RunloopBackend, spec: SandboxSpec) -> None:
        from runloop_api_client import Runloop

        self.spec = spec
        self._client = Runloop(bearer_token=backend.credential("RUNLOOP_API_KEY"))
        devbox = self._client.devboxes.create_and_await_running(
            name=f"nenyax-{uuid.uuid4().hex[:8]}",
            environment_variables=dict(spec.env),
            launch_parameters={"keep_alive_time_seconds": int(spec.timeout_s)},
        )
        self._id = devbox.id
        self.id = f"runloop-{devbox.id}"
        self.exec(f"mkdir -p {shlex.quote(spec.workdir)}", cwd="/")

    def exec(self, command, *, timeout_s=None, cwd=None, env=None) -> ExecResult:
        t0 = time.perf_counter()
        r = self._client.devboxes.execute_sync(
            self._id, command=_wrap(command, cwd or self.spec.workdir, env)
        )
        return _result(
            int(r.exit_status or 0), r.stdout or "", r.stderr or "", t0, self.spec.output_cap
        )

    def write_file(self, path: str, data: bytes | str) -> None:
        if isinstance(data, bytes):
            import base64

            target = _abs(self.spec, path)
            encoded = base64.b64encode(data).decode()
            quoted = shlex.quote(target)
            self.exec(
                f"mkdir -p $(dirname {quoted}) && echo {encoded} | base64 -d > {quoted}", cwd="/"
            )
            return
        self._client.devboxes.write_file_contents(
            self._id, file_path=_abs(self.spec, path), contents=data
        )

    def read_file(self, path: str) -> bytes:
        r = self.exec(f"base64 < {shlex.quote(_abs(self.spec, path))}", cwd="/")
        if not r.ok:
            raise FileNotFoundError(path)
        import base64

        return base64.b64decode(r.stdout)

    def destroy(self) -> None:
        self._client.devboxes.shutdown(self._id)


class RunloopBackend(SandboxBackend):
    name = "runloop"
    info = HOSTED
    credentials = ("RUNLOOP_API_KEY",)

    def missing(self) -> list[str]:
        return _need("runloop_api_client", "runloop_api_client") + super().missing()

    def create(self, spec: SandboxSpec | None = None) -> RunloopSandbox:
        return RunloopSandbox(self, spec or SandboxSpec())
