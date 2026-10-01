"""Run a platform job inside the stack's sandbox plug instead of the runner's own process.

The runner (outside) creates the sandbox, ships the Nenyax SDK into it, installs the
environment's driver, and runs ``python -m nenyax.sandboxed job.json`` there. The job writes its
events to ``events.jsonl`` (streamed back while it runs), its result to ``result.json`` and any
artifacts under ``out/``, which the runner copies out and publishes with the storage plug.

The runner's platform token only enters the sandbox when the environment itself must be
downloaded from the platform (a ``nenyax push`` package); untrusted environment code never sees it
otherwise.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import Any

#: Plugs that mean "run the environment in this sandbox backend"; anything else is in-process.
SANDBOX_PLUGS = {"process", "docker", "e2b", "daytona", "modal", "runloop"}
#: What each sandbox can give an environment that asks for it in its manifest.
PROVIDES: dict[str, set[str]] = {
    "nenyax": {"docker", "network", "gpu", "model_endpoint", "api_key"},  # the runner host
    "process": {"network", "model_endpoint", "api_key"},
    "docker": {"network", "model_endpoint", "api_key"},
    "e2b": {"network", "model_endpoint", "api_key"},
    "daytona": {"network", "model_endpoint", "api_key"},
    "modal": {"network", "model_endpoint", "api_key", "gpu"},
    "runloop": {"network", "model_endpoint", "api_key"},
}
DRIVER_PACKAGES = {
    "reasoning_gym": "reasoning-gym",
    "gymnasium": "gymnasium",
    "openenv": "openenv",
    "verifiers": "verifiers",
    "harbor": "harbor",
}
POLL_S = 2.0


def plug_of(config: dict[str, Any]) -> str:
    slot = (config.get("stack") or {}).get("sandbox") or {}
    return slot.get("plug") or "nenyax"


def check_requirements(requires: list[str] | None, plug: str, *, docker_here: bool = True) -> None:
    """Fail early, and clearly, when the environment needs something the sandbox can't give."""
    have = set(PROVIDES.get(plug, set()))
    if plug == "nenyax" and not docker_here:
        have.discard("docker")
    missing = [r for r in (requires or []) if r not in have]
    if missing:
        where = "this runner" if plug == "nenyax" else f"the {plug!r} sandbox"
        hint = (
            "run it on a runner with Docker (sandbox 'nenyax')"
            if "docker" in missing
            else "pick a sandbox that provides it"
        )
        need = ", ".join(missing)
        raise RuntimeError(f"this environment needs {need}, which {where} can't provide; {hint}")


def _sdk_tarball() -> bytes:
    """The installed Nenyax package, so the sandbox runs exactly this runner's version."""
    root = Path(__file__).resolve().parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for file in root.rglob("*.py"):
            tar.add(file, arcname=str(Path("nenyax") / file.relative_to(root)))
    return buf.getvalue()


def _driver(uri: str | None) -> str | None:
    name = (uri or "").split(":", 1)[0]
    return DRIVER_PACKAGES.get(name)


def _install(sb: Any, py: str, config: dict[str, Any]) -> None:
    sb.write_file("_nenyax.tar.gz", _sdk_tarball())
    r = sb.exec("mkdir -p _nenyax && tar -xzf _nenyax.tar.gz -C _nenyax && rm _nenyax.tar.gz")
    if not r.ok:
        raise RuntimeError(f"could not unpack the SDK in the sandbox: {r.stderr[-500:]}")
    wanted = [("pydantic", "pydantic>=2.6,<3")]
    if pkg := _driver((config.get("env") or {}).get("uri")):
        wanted.append((pkg.replace("-", "_"), pkg))
    for module, spec in wanted:
        if sb.exec(f"{py} -c 'import {module}'").ok:  # already there (e.g. the process plug)
            continue
        pip = f"{py} -m pip install --quiet '{spec}'"
        r = sb.exec(f"{pip} || {pip} --user", timeout_s=900)
        if not r.ok:
            tail = (r.stderr or r.stdout)[-800:]
            raise RuntimeError(f"installing {spec} in the sandbox failed: {tail}")


def execute_in_sandbox(
    kind: str,
    config: dict[str, Any],
    events: Any,
    *,
    env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``kind`` inside the stack's sandbox plug; returns the job's result (artifacts local)."""
    from . import sandbox
    from .remote import Cancelled

    plug = plug_of(config)
    check_requirements((config.get("env") or {}).get("requires"), plug)
    cfg = dict(((config.get("stack") or {}).get("sandbox") or {}).get("config") or {})
    backend = sandbox.get(plug, **{k: str(v) for k, v in cfg.get("options", {}).items()})
    spec = sandbox.SandboxSpec(
        image=cfg.get("image", "python:3.12-slim"),
        cpus=float(cfg.get("cpus", 2)),
        memory_mb=int(cfg.get("memory_mb", 4096)),
        timeout_s=float(cfg.get("timeout_s", 6 * 3600)),
        network=True,  # model APIs and hub downloads
        env={k: str(v) for k, v in (env_vars or {}).items()},
    )
    py = sys.executable if plug == "process" else "python3"
    events.emit("progress", {"stage": f"starting a {plug} sandbox"}, flush=True)
    with backend.create(spec) as sb:
        _install(sb, py, config)
        fetch = (config.get("env") or {}).get("fetch") or {}
        client = getattr(events, "client", None)
        inner = {**config, "stack": {**(config.get("stack") or {}), "sandbox": {"plug": "nenyax"}}}
        job = {
            "kind": kind,
            "config": inner,
            "run_id": getattr(events, "run_id", None),
            "platform": (
                {"url": client.url, "token": client.token}
                if client is not None and fetch.get("hub") == "nenyax"
                else None
            ),
        }
        sb.write_file("job.json", json.dumps(job))
        events.emit("progress", {"stage": f"running in the {plug} sandbox"}, flush=True)
        outcome: dict[str, Any] = {}
        thread = threading.Thread(
            target=lambda: outcome.setdefault(
                "exec", sb.exec(f"PYTHONPATH=_nenyax {py} -m nenyax.sandboxed job.json")
            ),
            daemon=True,
        )
        thread.start()
        seen = 0
        while thread.is_alive():
            thread.join(POLL_S)
            seen = _forward(sb, events, seen)
            if getattr(events, "cancelled", False):
                raise Cancelled()
        seen = _forward(sb, events, seen)
        result = outcome["exec"]
        try:
            out = json.loads(sb.read_file("result.json"))
        except Exception:  # noqa: BLE001 - the job crashed before writing a result
            tail = (result.stderr or result.stdout or "").strip()[-1500:]
            raise RuntimeError(f"the job failed inside the {plug} sandbox:\n{tail}") from None
        if out.get("error"):
            raise RuntimeError(out["error"])
        local = Path(tempfile.mkdtemp(prefix="nenyax-sandbox-out-"))
        for item in out.get("_artifacts") or []:
            target = local / Path(item["path"]).name
            target.write_bytes(sb.read_file(item["path"]))
            item["path"] = str(target)
        return out["result"] | {"_artifacts": out.get("_artifacts") or []}


def _forward(sb: Any, events: Any, seen: int) -> int:
    """Emit events the sandboxed job wrote since the last look."""
    try:
        lines = sb.read_file("events.jsonl").decode().splitlines()
    except Exception:  # noqa: BLE001 - not written yet
        return seen
    for line in lines[seen:]:
        with contextlib.suppress(ValueError):
            e = json.loads(line)
            events.emit(e["type"], e["payload"])
    events.flush()
    return len(lines)


# -- inside the sandbox --------------------------------------------------------------------------


class _FileEvents:
    def __init__(self, path: Path, run_id: int | None, client: Any) -> None:
        self.path, self.run_id, self.client = path, run_id, client
        self.cancelled = False
        self.lock = threading.Lock()

    def emit(self, type: str, payload: dict[str, Any], flush: bool = False) -> None:
        with self.lock, self.path.open("a") as f:
            f.write(json.dumps({"type": type, "payload": payload}, default=str) + "\n")

    def flush(self) -> None:
        pass


def _main(job_path: str) -> None:
    from .remote import PlatformClient, execute

    job = json.loads(Path(job_path).read_text())
    platform = job.get("platform") or {}
    client = PlatformClient(platform["url"], platform["token"]) if platform else None
    events = _FileEvents(Path("events.jsonl"), job.get("run_id"), client)
    out: dict[str, Any] = {}
    try:
        result = execute(job["kind"], job["config"], events, outdir=Path("out"))
        artifacts = result.pop("_artifacts", [])
        for item in artifacts:  # paths relative to the sandbox's working directory
            item["path"] = str(Path(item["path"]).resolve().relative_to(Path.cwd()))
        out = {"result": result, "_artifacts": artifacts}
    except Exception as e:  # noqa: BLE001 - report it to the runner outside
        out = {"error": f"{type(e).__name__}: {e}"[:4000]}
    Path("result.json").write_text(json.dumps(out, default=str))


if __name__ == "__main__":
    _main(sys.argv[1])
