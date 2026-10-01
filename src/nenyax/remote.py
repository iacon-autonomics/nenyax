"""The runner: execute platform jobs on your own compute.

    nenyax runner --url https://<platform>/api --token nxr_...

The runner connects *outbound* only. It claims jobs, runs them with this machine's Nenyax install,
streams progress, and reports results. Model keys, data and environments never leave the machine
unless a job explicitly exports them; the platform only ever sees scores, metrics and replays.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any

from . import __version__

log = logging.getLogger("nenyax.runner")

REPLAY_CHARS = 6_000  # keep replays readable and payloads small


class PlatformClient:
    def __init__(self, url: str, token: str, timeout_s: float = 60.0) -> None:
        self.url, self.token, self.timeout_s = url.rstrip("/"), token, timeout_s

    def call(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read() or b"{}")


def machine_info() -> dict[str, Any]:
    """What this runner can do, so the platform can route jobs to capable machines."""
    from .registry import drivers

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "nenyax": __version__,
        "docker": bool(shutil.which("docker")),
        "drivers": sorted(n for n, d in drivers().items() if d.available()),
    }
    try:
        import torch

        info["gpu"] = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else None
        )
    except ImportError:
        info["gpu"] = None
    return info


class _Events:
    """Buffers events and flushes them in batches; remembers whether the run was cancelled."""

    def __init__(self, client: PlatformClient, run_id: int) -> None:
        self.client, self.run_id = client, run_id
        self.buffer: list[dict[str, Any]] = []
        self.cancelled = False
        self.lock = threading.Lock()

    def emit(self, type: str, payload: dict[str, Any], flush: bool = False) -> None:
        with self.lock:
            self.buffer.append({"type": type, "payload": payload})
        if flush or len(self.buffer) >= 10:
            self.flush()

    def flush(self) -> None:
        with self.lock:
            batch, self.buffer = self.buffer, []
        if not batch:
            return
        try:
            reply = self.client.call(f"/runner/runs/{self.run_id}/events", {"events": batch})
            self.cancelled = self.cancelled or bool(reply.get("cancel"))
        except (urllib.error.URLError, OSError) as e:
            log.warning("could not send events: %s", e)


class Cancelled(Exception):
    pass


@contextlib.contextmanager
def job_env(env: dict[str, str] | None):
    """Set a job's connection env vars (provider keys, hub tokens) only while it runs."""
    saved = {k: os.environ.get(k) for k in env or {}}
    os.environ.update({k: str(v) for k, v in (env or {}).items()})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def install(spec: str, events: _Events) -> None:
    """Install a custom component (a pip or git spec) into this runner's interpreter."""
    events.emit("log", {"level": "info", "message": f"installing {spec}"}, flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", spec],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pip install {spec} failed:\n{(proc.stderr or proc.stdout)[-2000:]}")
    events.emit("log", {"level": "info", "message": f"installed {spec}"}, flush=True)


def _episode(traj: Any, index: int) -> dict[str, Any]:
    from .replay import render

    return {
        "index": index,
        "task": traj.task.id if traj.task else None,
        "seed": traj.seed,
        "score": traj.score,
        "ok": traj.ok,
        "error": traj.error,
        "steps": len(traj.steps),
        "model_calls": len(traj.model_calls),
        "duration_s": round(traj.duration_s or 0.0, 3),
        "replay": render(traj)[:REPLAY_CHARS],
    }


def _load_env(
    spec: dict[str, Any], max_tasks: int | None, client: PlatformClient | None = None
) -> Any:
    from . import integrations
    from .registry import load

    fetch = spec.get("fetch")
    if fetch and fetch.get("hub") == "nenyax":  # pushed to the platform with `nenyax push`
        if client is None:
            raise RuntimeError("platform packages load through a runner connected to the platform")
        from .platform import load_package

        return load_package(fetch, client.url, client.token)
    if fetch:  # hub listings: install or download on this machine first
        hub = integrations.hubs()[fetch["hub"]]
        wanted = fetch["name"]
        query = wanted.split("@")[0].split("/")[-1]
        hits = [h for h in hub.search(query, limit=50) if h.name == wanted]
        if not hits:
            raise RuntimeError(f"{wanted} not found on {fetch['hub']}")
        options = {"max_tasks": max_tasks} if fetch["hub"] == "harbor" else {}
        return load(hub.fetch(hits[0], **options))
    return load(spec["uri"])


def _policy(env: Any, model: dict[str, Any] | None, task: Any = None) -> Any:
    if model:
        from .config import build

        return build("model", dict(model))
    for make in (
        lambda: env.reference_policy(task),
        lambda: getattr(env, "random_policy", lambda **_: None)(seed=0),
    ):
        policy = make()
        if policy is not None:
            return policy
    raise RuntimeError("this environment has no reference policy; choose a model")


def execute(kind: str, config: dict[str, Any], events: _Events) -> dict[str, Any]:
    """Run one job. Pure Nenyax SDK calls; the same thing works from a script."""
    from . import check as conformance
    from .config import build
    from .train import train

    params = config.get("params") or {}
    events.emit("progress", {"stage": "loading environment"}, flush=True)
    env = _load_env(
        config["env"],
        max_tasks=int(params.get("max_tasks", 3)),
        client=getattr(events, "client", None),
    )
    events.emit(
        "progress",
        {"stage": "loaded", "env": env.id, "mode": env.mode.value, "tasks": env.manifest.num_tasks},
        flush=True,
    )

    if kind == "try":
        tasks = env.tasks(limit=1)
        task = tasks[0] if tasks else None
        traj = env.rollout(_policy(env, config.get("model"), task), task=task, seed=0)
        events.emit("episode", _episode(traj, 0), flush=True)
        if traj.error:
            raise RuntimeError(traj.error)
        return {"score": traj.score, "episodes": 1}

    if kind == "baseline":
        n = int(params.get("n", 20))
        model = config.get("model")
        scores = []
        tasks = env.tasks(limit=n) or [None] * n
        for i, task in enumerate(tasks[:n]):
            if events.cancelled:
                raise Cancelled()
            traj = env.rollout(_policy(env, model, task), task=task, seed=i)
            scores.append(traj.score or 0.0)
            events.emit("episode", _episode(traj, i))
        events.flush()
        return {"score": sum(scores) / len(scores) if scores else 0.0, "episodes": len(scores)}

    if kind == "check":
        report = conformance(
            env,
            samples=int(params.get("samples", 2)),
            on_check=lambda c: events.emit(
                "metric",
                {
                    "check": c.name,
                    "tier": c.tier.label,
                    "status": c.status,
                    "detail": c.detail,
                    "duration_s": round(c.duration_s, 3),
                },
                flush=True,
            ),
        )
        return {
            "tier": report.tier.label,
            "checks": [
                {"name": c.name, "tier": c.tier.label, "status": c.status, "detail": c.detail}
                for c in report.checks
            ],
        }

    if kind == "train":
        learner_spec = dict(config.get("learner") or {"use": "incontext"})
        if spec := learner_spec.pop("install", None):  # your own algorithm, from pip or git
            install(spec, events)
        if learner_spec.get("use", "incontext") == "incontext":
            learner_spec.setdefault("model", config.get("model"))
        elif "model" not in learner_spec and config.get("model"):
            learner_spec["model"] = config["model"]["name"]  # weight learners take a model id
        learner = build("learner", learner_spec)

        def on_round(r: Any) -> None:
            events.emit("round", asdict(r), flush=True)
            if events.cancelled:
                raise Cancelled()

        result = train(
            env,
            learner,
            rounds=int(params.get("rounds", 3)),
            group_size=int(params.get("group_size", 4)),
            batch=int(params.get("batch", 6)),
            concurrency=int(params.get("concurrency", 1)),
            on_round=on_round,
            callbacks=[_EpisodeForwarder(events)],
        )
        per_round = int(params.get("batch", 6)) * int(params.get("group_size", 4))
        return {
            "baseline": result.baseline,
            "best": result.best,
            "score": result.best,
            "rounds": len(result.history),
            "episodes": per_round * len(result.history),
        }

    raise ValueError(f"unknown run kind {kind!r}")


class _EpisodeForwarder:
    """Telemetry callback that streams a sample of training episodes."""

    def __init__(self, events: _Events, every: int = 4) -> None:
        self.events, self.every, self.seen = events, every, 0

    def on_trajectory(self, traj: Any) -> None:
        if self.seen % self.every == 0:
            self.events.emit("episode", _episode(traj, self.seen))
        self.seen += 1


def _beat(client: PlatformClient, info: dict[str, Any], stop: threading.Event) -> None:
    """Keep the runner marked online during long jobs."""
    while not stop.wait(10):
        with contextlib.suppress(urllib.error.URLError, OSError):
            client.call("/runner/heartbeat", {"info": info})


def serve(url: str, token: str, *, poll_s: float = 2.0, once: bool = False) -> None:
    """Heartbeat, claim, execute, report. Runs until interrupted (or after one job with once)."""
    client = PlatformClient(url, token)
    info = machine_info()
    hello = client.call("/runner/heartbeat", {"info": info})
    log.info("connected as %s (%s)", hello.get("runner"), url)
    last_beat = time.monotonic()
    while True:
        if time.monotonic() - last_beat > 10:
            client.call("/runner/heartbeat", {"info": info})
            last_beat = time.monotonic()
        try:
            job = client.call("/runner/claim").get("run")
        except (urllib.error.URLError, OSError) as e:
            log.warning("platform unreachable: %s", e)
            time.sleep(poll_s * 2)
            continue
        if not job:
            if once:
                return
            time.sleep(poll_s)
            continue
        run_id = job["id"]
        log.info("run %s: %s", run_id, job["kind"])
        events = _Events(client, run_id)
        stop_beat = threading.Event()

        threading.Thread(target=_beat, args=(client, info, stop_beat), daemon=True).start()
        try:
            with job_env(job.get("env")):
                result = execute(job["kind"], job["config"], events)
            events.flush()
            client.call(f"/runner/runs/{run_id}/finish", {"status": "succeeded", "result": result})
        except Cancelled:
            events.flush()
        except Exception as e:  # report every failure to the platform; keep serving
            events.emit("log", {"level": "error", "message": f"{type(e).__name__}: {e}"})
            events.flush()
            client.call(
                f"/runner/runs/{run_id}/finish",
                {"status": "failed", "error": f"{type(e).__name__}: {e}"[:4000]},
            )
        finally:
            stop_beat.set()
        if once:
            return
