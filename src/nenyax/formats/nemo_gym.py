"""NVIDIA NeMo Gym driver: talk to a running *resources server* over HTTP.

A NeMo Gym resources server owns tasks and verification. Its contract is small and stable::

    POST /seed_session  {}                         -> per-episode state (cookie session)
    POST /verify        {**task_row, "response": <OpenAI Responses object>} -> {"reward": ...}

Nenyax drives single-turn resources servers (math, MCQA, reasoning-gym, instruction following,
...) in step mode: the observation is the row's ``responses_create_params.input`` rendered as
chat messages, the action is the assistant's text, and ``/verify`` is the judge.

Because the driver speaks HTTP only, NeMo Gym's own Python (>=3.13.14), Ray and per-server
virtualenvs never enter your process. Start a server with NeMo Gym's tooling (``ng_run``) or with
``tools/nemo_gym_serve.py`` from this repository, then::

    nenyax.load("nemo_gym:http://127.0.0.1:8000?data=path/to/tasks.jsonl")

Multi-turn and tool-calling servers are driven by NeMo Gym *agent* servers; that path is not
implemented yet and such rows are rejected rather than silently mis-scored.
"""

from __future__ import annotations

import http.cookiejar
import json
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .._util import to_json
from ..environment import Session, StepEnvironment
from ..registry import Driver
from ..types import (
    JSON,
    Capabilities,
    Judgment,
    Manifest,
    Message,
    Mode,
    Observation,
    Step,
    TaskRef,
)


def responses_object(text: str, model: str = "nenyax") -> dict[str, Any]:
    """Wrap assistant text as a minimal OpenAI Responses API ``Response`` object."""
    return {
        "id": f"resp_{uuid.uuid4().hex[:12]}",
        "created_at": int(time.time()),
        "model": model,
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def _input_messages(params: dict[str, Any]) -> list[Message]:
    raw = params.get("input", [])
    if isinstance(raw, str):
        raw = [{"role": "user", "content": raw}]
    messages = []
    if params.get("instructions"):
        messages.append(Message(role="system", content=params["instructions"]))
    for item in raw:
        role = item.get("role", "user")
        content = item.get("content")
        if isinstance(content, list):  # Responses content parts -> plain text
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        messages.append(Message(role="system" if role == "developer" else role, content=content))
    return messages


class _Client:
    """A cookie-keeping JSON client: NeMo Gym keys per-episode state off the session cookie."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener.open(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())


class NemoGymSession(Session):
    def __init__(self, env: NemoGymEnvironment, row: dict[str, Any], task_id: str) -> None:
        self.env, self.row, self.task_id = env, row, task_id
        self.client = _Client(env.base_url, env.timeout_s)
        self.client.post("/seed_session", {})
        self.history: list[Step] = []
        self.verdict: dict[str, Any] = {}
        self.done = False
        messages = _input_messages(row["responses_create_params"])
        self.observation = Observation(
            data=to_json(row["responses_create_params"]),
            text=messages[-1].content if messages else "",
            messages=messages,
            info={"task_id": task_id},
        )

    def act(self, action: JSON) -> Step:
        if self.done:
            raise RuntimeError("episode is over; open a new session")
        text = action if isinstance(action, str) else json.dumps(action)
        self.verdict = self.client.post("/verify", {**self.row, "response": responses_object(text)})
        reward = float(self.verdict["reward"])
        step = Step(
            action=text,
            observation=Observation(
                text="", info={"failure_reason": self.verdict.get("failure_reason")}
            ),
            reward=reward,
            terminated=True,
        )
        self.history.append(step)
        self.done = True
        return step

    def judge(self) -> Judgment:
        extras = {
            k: to_json(v)
            for k, v in self.verdict.items()
            if k not in self.row and k not in ("response", "reward")
        }
        score = self.history[-1].reward if self.history else 0.0
        return Judgment(
            score=score,
            passed=score >= 1.0,
            source="verifier",
            judge=f"nemo_gym:{self.env.server_name}/verify",
            details=extras,
        )


class NemoGymEnvironment(StepEnvironment):
    default_max_steps = 1

    def __init__(
        self,
        base_url: str,
        *,
        data: str,
        name: str | None = None,
        reference: str | None = None,
        null: str = "",
        timeout_s: float = 300.0,
    ) -> None:
        """
        Args:
            base_url: URL of a running resources server.
            data: JSONL file of task rows in the server's format.
            reference: optional Python format string over a row that yields a correct answer,
                e.g. ``"\\\\boxed{{{expected_answer}}}"``; enables the audit tier.
            null: answer expected to score zero, used by audits.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.rows = [
            json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()
        ]
        for i, row in enumerate(self.rows):
            params = row.get("responses_create_params") or {}
            if params.get("tools"):
                raise ValueError(
                    f"row {i} uses tools; tool-calling NeMo Gym servers need an agent server, "
                    "which this driver does not drive yet"
                )
        self.server_name = name or Path(data).parent.parent.name or "resources"
        self.reference_template = reference
        self.null_answer = null
        self.manifest = Manifest(
            id=f"nemo_gym:{self.base_url}#{self.server_name}",
            name=self.server_name,
            source_format="nemo_gym",
            mode=Mode.STEP,
            capabilities=Capabilities(
                resettable=True, text=True, reward_timing="terminal", requires=["network"]
            ),
            action_schema={"type": "string"},
            num_tasks=len(self.rows),
            extra={"protocol": "resources_server/verify", "data": str(Path(data).resolve())},
        )

    def tasks(self, split: str | None = None, limit: int | None = None) -> list[TaskRef]:
        return [TaskRef(id=str(i)) for i in range(len(self.rows))][:limit]

    def session(self, *, task: TaskRef | None = None, seed: int | None = None) -> NemoGymSession:
        index = int(task.id) if task else (seed or 0) % len(self.rows)
        return NemoGymSession(self, self.rows[index], str(index))

    def reference_policy(self, task: TaskRef | None = None):
        if self.reference_template is None:
            return None
        rows, template = self.rows, self.reference_template
        return lambda obs: template.format(**rows[int(obs.info["task_id"])])

    def null_policy(self, task: TaskRef | None = None):
        answer = self.null_answer
        return lambda obs: answer


class NemoGymDriver(Driver):
    name = "nemo_gym"
    format = "NeMo Gym resources server (NVIDIA)"
    module = None  # HTTP only; NeMo Gym itself runs in its own environment

    def load(self, target: str, **options: Any) -> NemoGymEnvironment:
        if "data" not in options:
            raise ValueError("nemo_gym needs ?data=<tasks.jsonl>")
        return NemoGymEnvironment(target, **options)
