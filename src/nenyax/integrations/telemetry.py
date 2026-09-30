"""Telemetry: observe episodes without changing how they run.

A callback is any object with ``on_trajectory(traj)``; pass callbacks to :func:`nenyax.run`::

    nenyax.run(env, policy, n=100, callbacks=[OpenTelemetry(), MLflow(experiment="rl")])

Nothing is sent anywhere unless you add a callback. Built in: OpenTelemetry (GenAI semantic
conventions, so Langfuse, Arize Phoenix, Jaeger, Honeycomb and friends can ingest it), MLflow and
Weights & Biases. More register through the ``nenyax.telemetry`` entry-point group.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..types import Trajectory
from .base import Integration


@runtime_checkable
class Callback(Protocol):
    def on_trajectory(self, traj: Trajectory) -> None: ...


def episode_metrics(traj: Trajectory) -> dict[str, float]:
    """Flat numeric summary of one episode, shared by every metrics backend."""
    metrics = {
        "score": traj.score if traj.score is not None else float("nan"),
        "ok": float(traj.ok),
        "steps": float(len(traj.steps)),
        "model_calls": float(len(traj.model_calls)),
        "duration_s": traj.duration_s or 0.0,
    }
    if traj.judgment:
        metrics.update({f"reward/{k}": v for k, v in traj.judgment.components.items()})
    return metrics


class OpenTelemetry(Integration):
    """One span per episode, one child span per model call (``gen_ai.*`` attributes)."""

    name, kind, modules = "opentelemetry", "telemetry", ("opentelemetry",)
    summary = "OpenTelemetry traces (Langfuse, Phoenix, Jaeger, ... via OTLP)"

    def __init__(self, tracer: Any = None, **credentials: str) -> None:
        super().__init__(**credentials)
        from opentelemetry import trace

        self.tracer = tracer or trace.get_tracer("nenyax")

    def on_trajectory(self, traj: Trajectory) -> None:
        start = int(traj.started_at * 1e9)
        end = start + int((traj.duration_s or 0) * 1e9)
        attrs = {
            "nenyax.env_id": traj.env_id,
            "nenyax.mode": traj.mode.value,
            "nenyax.task_id": traj.task.id if traj.task else "",
            "nenyax.seed": traj.seed if traj.seed is not None else -1,
            "nenyax.score": traj.score if traj.score is not None else float("nan"),
            "nenyax.ok": traj.ok,
        }
        with self.tracer.start_as_current_span(
            "nenyax.episode", start_time=start, attributes=attrs, end_on_exit=False
        ) as span:
            for call in traj.model_calls:
                model = str(call.request.get("model", ""))
                usage = (call.response or {}).get("usage") or {}
                child = self.tracer.start_span(
                    f"chat {model}",
                    attributes={
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": model,
                        "gen_ai.usage.input_tokens": usage.get("prompt_tokens", 0),
                        "gen_ai.usage.output_tokens": usage.get("completion_tokens", 0),
                    },
                )
                if call.error:
                    child.set_attribute("error.type", call.error[:200])
                child.end()
            if traj.error:
                span.set_attribute("error.type", traj.error[:200])
            span.end(end_time=end)


class MLflow(Integration):
    name, kind, modules = "mlflow", "telemetry", ("mlflow",)
    summary = "MLflow run metrics (local or tracking server)"

    def __init__(
        self, experiment: str = "nenyax", run_name: str | None = None, **credentials: str
    ) -> None:
        super().__init__(**credentials)
        import mlflow

        self.mlflow = mlflow
        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run(run_name=run_name)
        self.step = 0

    def on_trajectory(self, traj: Trajectory) -> None:
        metrics = {k.replace("/", "_"): v for k, v in episode_metrics(traj).items()}
        self.mlflow.log_metrics(metrics, step=self.step, run_id=self.run.info.run_id)
        self.step += 1

    def close(self) -> None:
        self.mlflow.end_run()


class WandB(Integration):
    name, kind, modules = "wandb", "telemetry", ("wandb",)
    credentials = ()  # WANDB_API_KEY for online runs; mode="offline" needs none
    summary = "Weights & Biases (online with WANDB_API_KEY, or offline)"

    def __init__(self, project: str = "nenyax", mode: str | None = None, **credentials: str):
        super().__init__(**credentials)
        import wandb

        self.run = wandb.init(
            project=project,
            mode=mode or ("online" if self.credential("WANDB_API_KEY") else "offline"),
        )

    def on_trajectory(self, traj: Trajectory) -> None:
        self.run.log(episode_metrics(traj))

    def close(self) -> None:
        self.run.finish()
